#!/usr/bin/env python3
# semantic_chef.py - publish short announce on /robot_speech and detailed text only on /chef_suggestions
import rospy
import os
import threading
import time
import traceback
from std_msgs.msg import String
from agents import llm_call
import json

DEFAULT_TIMEOUT_SEC = float(os.getenv("AZURE_TIMEOUT_SEC", 60.0))
DEFAULT_MAX_RECIPES = int(os.getenv("MAX_RECIPES", 3))
DEFAULT_DEDUP_WINDOW = float(os.getenv("CHEF_DEDUP_WINDOW", 6.0))
DEBUG_ENV = os.getenv("SEMANTIC_CHEF_DEBUG", "0") in ("1", "true", "True")

# NOTE: FOOD_SET used only for heuristic token detection (not exhaustive)
FOOD_SET = {
    'uova','uovo','pasta','spaghetti','salsiccia','salsicce','wurstel','formaggio','pepe','sale','pomodoro','latte','pane','farina',
    'burro','olio','pollo','manzo','carne','pesce','riso','patate','cipolla','aglio','yogurt','miele','zucchero',
    'mozzarella','parmigiano','pomodori','carote','insalata','lattuga','cetriolo','zucchine',
    'egg','eggs','sausage','cheese','pepper','salt','tomato','milk','bread','flour','butter','oil',
    'chicken','beef','meat','fish','rice','potato','onion','garlic','yogurt','honey','sugar','carrot','lettuce'
}

MEAT_TOKENS = {'pollo','manzo','carne','salsiccia','salsicce','wurstel','prosciutto','bacon','pancetta','maiale','vitello','fish','pesce','chicken','beef','sausage','pork','bacon'}

def logd(*args):
    if DEBUG_ENV:
        rospy.loginfo("[semantic_chef DEBUG] " + " ".join(str(a) for a in args))

def is_food_token(tok: str) -> bool:
    if not tok:
        return False
    t = tok.strip().lower()
    if t in FOOD_SET:
        return True
    for f in FOOD_SET:
        if f in t:
            return True
    return False

def simple_local_fallback(ingredients_list, max_recipes=3):
    lower = set([i.strip().lower() for i in ingredients_list if i.strip()])
    suggestions = []
    if 'pasta' in lower or 'spaghetti' in lower:
        if 'uova' in lower and 'formaggio' in lower:
            suggestions.append(("Carbonara-approssimativa",
                                "Ingredienti: pasta, uova, formaggio. Procedura: cuoci la pasta; mescola uova+formaggio; unisci tutto. Tempo ~15-25 min."))
        elif 'salsiccia' in lower or 'pomodoro' in lower:
            suggestions.append(("Pasta con condimento veloce",
                                "Ingredienti: pasta + salsiccia/pomodoro. Procedura: rosola salsiccia, aggiungi pomodoro, mescola con pasta. Tempo ~20-30 min."))
    if 'uova' in lower and 'farina' in lower and 'latte' in lower:
        suggestions.append(("Pancake veloce",
                            "Ingredienti: farina, uova, latte. Procedura: mescola, cuoci in padella a fiamma media. Tempo ~15-20 min."))
    if 'pane' in lower and 'formaggio' in lower:
        suggestions.append(("Toast al formaggio",
                            "Ingredienti: pane, formaggio. Procedura: farcisci e tosta. Tempo ~5-10 min."))
    if not suggestions:
        if lower:
            items = ", ".join(list(lower)[:6])
            suggestions.append(("Ricetta con ingredienti disponibili",
                                f"Ingredienti: {items}. Procedura: combina ingredienti semplici, cuoci al forno o in padella. Tempo ~20-40 min."))
        else:
            suggestions.append(("Nessun ingrediente specificato", "Non ci sono ingredienti: suggerisci di fornire una lista."))
    return suggestions[:max_recipes]

class SemanticChefNode:
    def __init__(self):
        rospy.init_node("semantic_chef", anonymous=False)
        self.TIMEOUT_SEC = float(rospy.get_param("~azure_timeout_sec", DEFAULT_TIMEOUT_SEC))
        self.MAX_RECIPES = int(rospy.get_param("~max_recipes", DEFAULT_MAX_RECIPES))
        self.DEDUP_WINDOW = float(rospy.get_param("~dedup_window", DEFAULT_DEDUP_WINDOW))
        self.DEBUG = rospy.get_param("~debug", DEBUG_ENV)

        # topics
        self.pub = rospy.Publisher("/chef_suggestions", String, queue_size=2)  # detailed text
        self.pub_status = rospy.Publisher("/chef_status", String, queue_size=2)
        self.pub_robot_speech = rospy.Publisher("/robot_speech", String, queue_size=4)  # short announce only

        # locks / dedup
        self._processing_lock = threading.Lock()
        self._last_processed_csv = None
        self._last_processed_time = 0.0

        # Keep track of the most recent structured request CSV to avoid duplicate processing:
        self._last_structured_csv = None
        self._last_structured_time = 0.0
        self._structured_ignore_window = 5.0  # seconds: ignore CSVs that match a recent structured request

        rospy.Subscriber("/detected_ingredients", String, self.cb_ingredients, queue_size=1)
        rospy.Subscriber("/dispensa_ingredients", String, self.cb_ingredients, queue_size=1)
        rospy.Subscriber("/chef_request", String, self.cb_chef_request, queue_size=1)

        rospy.loginfo("semantic_chef started. timeout=%.1f dedup=%.1fs", self.TIMEOUT_SEC, self.DEDUP_WINDOW)

    def report_status(self, s: str):
        try:
            self.pub_status.publish(String(data=s))
        except Exception:
            pass
        rospy.loginfo("[semantic_chef] %s", s)

    def _should_ignore_duplicate(self, csv_str: str) -> bool:
        if not csv_str:
            return True
        now = time.time()
        if self._last_processed_csv == csv_str and (now - self._last_processed_time) <= self.DEDUP_WINDOW:
            logd("Ignoring duplicate CSV (within dedup window):", csv_str)
            return True
        return False

    def cb_ingredients(self, msg: String):
        """
        Legacy handler for CSV on /detected_ingredients.

        If a structured request with the same CSV arrived recently, ignore this one
        to avoid processing twice (structured handler is preferred).
        """
        try:
            csv = msg.data.strip() if msg and msg.data else ""
            rospy.loginfo("SemanticChef: raw received ingredients (CSV): %s", csv)
            if not csv:
                self.report_status("received_empty")
                return

            # If same csv was just handled as structured request, ignore this CSV
            now = time.time()
            if self._last_structured_csv and self._last_structured_csv == csv and (now - self._last_structured_time) <= self._structured_ignore_window:
                rospy.loginfo("Ignoring CSV because a structured request with same CSV was received recently.")
                self.report_status("structured_taken")
                return

            if self._should_ignore_duplicate(csv):
                self.report_status("duplicate_ignored")
                return

            if not self._processing_lock.acquire(blocking=False):
                rospy.logwarn("SemanticChef busy: another request is being processed. Ignoring this one.")
                self.report_status("busy_ignored")
                return

            try:
                items = [i.strip() for i in csv.split(",") if i.strip()]
                foods = [i for i in items if is_food_token(i)]
                if not foods:
                    rospy.logwarn("No food items after filtering (detected non-food). Items: %s", items)
                    self.report_status("no_food_detected")
                    out_text = "Nessun alimento valido rilevato (filtraggio). Per favore inserisci ingredienti reali."
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data=out_text))
                    except Exception:
                        pass
                    self._last_processed_csv = csv
                    self._last_processed_time = time.time()
                    return

                rospy.loginfo("SemanticChef: filtered food ingredients (CSV flow): %s", foods)
                self.report_status("ingredients_accepted")
                self._last_processed_csv = csv
                self._last_processed_time = time.time()

                # worker LLM (legacy, no dish_type/dietary)
                result = {'text': None, 'error': None}
                worker = threading.Thread(target=self._llm_worker, args=(foods, result), daemon=True)
                start = time.time()
                worker.start()
                worker.join(timeout=self.TIMEOUT_SEC)
                elapsed = time.time() - start

                if worker.is_alive():
                    rospy.logwarn("LLM call timed out after %.1f s -> fallback", elapsed)
                    self.report_status("timeout_fallback")
                    suggestions = simple_local_fallback(foods, max_recipes=self.MAX_RECIPES)
                    out_text = self._format_suggestions_text(suggestions, source="fallback:timeout")
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data="Ho usato una strategia alternativa per generare ricette (timeout). Controlla i dettagli nel terminale."))
                    except Exception:
                        pass
                    return

                if result.get('error'):
                    rospy.logwarn("LLM error: %s -> fallback", result.get('error'))
                    self.report_status("error_fallback")
                    suggestions = simple_local_fallback(foods, max_recipes=self.MAX_RECIPES)
                    out_text = self._format_suggestions_text(suggestions, source="fallback:error")
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data="Si è verificato un errore nel servizio linguistico: uso una strategia alternativa."))
                    except Exception:
                        pass
                    return

                llm_text = result.get('text', "").strip()
                if not llm_text:
                    rospy.logwarn("LLM returned empty -> fallback")
                    self.report_status("empty_fallback")
                    suggestions = simple_local_fallback(foods, max_recipes=self.MAX_RECIPES)
                    out_text = self._format_suggestions_text(suggestions, source="fallback:empty")
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data="Il servizio linguistico ha risposto vuoto. Ho generato suggerimenti locali."))
                    except Exception:
                        pass
                    return

                elapsed = time.time() - start
                rospy.loginfo("LLM returned suggestions in %.1f s (CSV flow)", elapsed)
                self.report_status("suggestions_ready")
                try:
                    self.pub.publish(String(data=llm_text))
                    self.pub_robot_speech.publish(String(data="Ho trovato delle ricette basate sugli ingredienti forniti. Controlla il terminale per i dettagli."))
                except Exception as e:
                    rospy.logwarn("Failed publishing suggestions: %s", str(e))

            finally:
                try:
                    self._processing_lock.release()
                except Exception:
                    pass

        except Exception as e:
            rospy.logerr("Unhandled error in cb_ingredients: %s\n%s", str(e), traceback.format_exc())
            self.report_status("exception_fallback")
            suggestions = simple_local_fallback([], max_recipes=self.MAX_RECIPES)
            out_text = self._format_suggestions_text(suggestions, source="fallback:exception")
            try:
                self.pub.publish(String(data=out_text))
                self.pub_robot_speech.publish(String(data="Si è verificato un errore interno: ho applicato una strategia alternativa per le ricette."))
            except Exception:
                pass

    def cb_chef_request(self, msg: String):
        """
        Handler per richieste strutturate dal dispensa_inspector.
        Supporta sia JSON che CSV (legacy).
        JSON atteso:
           {"ingredients": [...], "dish_type": "...", "dietary": "..."}

        If JSON -> structured flow (preferred).
        If CSV -> fallback to cb_ingredients.
        """
        try:
            raw = msg.data.strip() if msg and msg.data else ""
            rospy.loginfo("cb_chef_request received raw: %s", raw[:300])
            if not raw:
                self.report_status("chef_request_empty")
                return

            # Try parse JSON
            payload = None
            try:
                payload = json.loads(raw)
            except Exception:
                payload = None

            if payload and isinstance(payload, dict) and payload.get("ingredients"):
                items = [i.strip() for i in payload.get("ingredients") if str(i).strip()]
                dish_type = str(payload.get("dish_type", "")).strip()
                dietary = str(payload.get("dietary", "")).strip()
            else:
                # assume legacy CSV -> reuse cb_ingredients behaviour
                rospy.loginfo("chef_request payload not JSON or missing ingredients -> treating as CSV")
                tmp = String(data=raw)
                self.cb_ingredients(tmp)
                return

            csv = ",".join(items)
            # register this structured csv so that subsequent CSV callback is ignored briefly
            self._last_structured_csv = csv
            self._last_structured_time = time.time()

            if self._should_ignore_duplicate(csv):
                self.report_status("duplicate_ignored")
                return

            if not self._processing_lock.acquire(blocking=False):
                rospy.logwarn("SemanticChef busy: another request is being processed. Ignoring this structured request.")
                self.report_status("busy_ignored")
                return

            try:
                # filter food tokens
                foods = [i for i in items if is_food_token(i)]
                if not foods:
                    rospy.logwarn("No food items after filtering (structured). Items: %s", items)
                    self.report_status("no_food_detected")
                    out_text = "Nessun alimento valido rilevato (filtraggio). Per favore inserisci ingredienti reali."
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data=out_text))
                    except Exception:
                        pass
                    self._last_processed_csv = csv
                    self._last_processed_time = time.time()
                    return

                # Apply simple dietary filters locally (heuristic)
                dietary_lower = dietary.lower() if dietary else ""
                effective_foods = list(foods)  # shallow copy
                if dietary_lower:
                    if "vegetar" in dietary_lower:
                        # remove obvious meat tokens
                        new_foods = []
                        for tok in effective_foods:
                            low = tok.lower()
                            if any(mt in low for mt in MEAT_TOKENS):
                                logd("Removing token due to vegetarian preference:", tok)
                                continue
                            new_foods.append(tok)
                        effective_foods = new_foods
                    # other heuristics can be added here, e.g. "no lattosio" -> tag in prompt

                # If after applying dietary filters we have no foods left, fallback to notifying user
                if not effective_foods:
                    rospy.logwarn("After applying dietary filters no food items remain. Aborting structured request.")
                    self.report_status("no_food_after_dietary")
                    out_text = "Le restrizioni specificate escludono tutti gli ingredienti disponibili. Per favore rimuovi o modifica le restrizioni."
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data=out_text))
                    except Exception:
                        pass
                    self._last_processed_csv = csv
                    self._last_processed_time = time.time()
                    return

                rospy.loginfo("SemanticChef (structured): filtered food ingredients before LLM: %s (dietary='%s', dish_type='%s')",
                              effective_foods, dietary, dish_type)
                self.report_status("ingredients_accepted")
                self._last_processed_csv = csv
                self._last_processed_time = time.time()

                # worker LLM (structured)
                result = {'text': None, 'error': None}
                worker = threading.Thread(target=self._llm_worker_structured, args=(effective_foods, dish_type, dietary, result), daemon=True)
                start = time.time()
                worker.start()
                worker.join(timeout=self.TIMEOUT_SEC)
                elapsed = time.time() - start

                if worker.is_alive():
                    rospy.logwarn("LLM call timed out after %.1f s -> fallback", elapsed)
                    self.report_status("timeout_fallback")
                    suggestions = simple_local_fallback(effective_foods, max_recipes=self.MAX_RECIPES)
                    out_text = self._format_suggestions_text(suggestions, source="fallback:timeout")
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data="Ho usato una strategia alternativa per generare ricette (timeout). Controlla il terminale per i dettagli."))
                    except Exception:
                        pass
                    return

                if result.get('error'):
                    rospy.logwarn("LLM error: %s -> fallback", result.get('error'))
                    self.report_status("error_fallback")
                    suggestions = simple_local_fallback(effective_foods, max_recipes=self.MAX_RECIPES)
                    out_text = self._format_suggestions_text(suggestions, source="fallback:error")
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data="Si è verificato un errore nel servizio linguistico: uso una strategia alternativa."))
                    except Exception:
                        pass
                    return

                llm_text = result.get('text', "").strip()
                if not llm_text:
                    rospy.logwarn("LLM returned empty -> fallback")
                    self.report_status("empty_fallback")
                    suggestions = simple_local_fallback(effective_foods, max_recipes=self.MAX_RECIPES)
                    out_text = self._format_suggestions_text(suggestions, source="fallback:empty")
                    self.pub.publish(String(data=out_text))
                    try:
                        self.pub_robot_speech.publish(String(data="Il servizio linguistico ha risposto vuoto. Ho generato suggerimenti locali."))
                    except Exception:
                        pass
                    return

                rospy.loginfo("LLM returned suggestions (structured).")
                self.report_status("suggestions_ready")
                try:
                    self.pub.publish(String(data=llm_text))
                    self.pub_robot_speech.publish(String(data="Ho trovato delle ricette basate sugli ingredienti e le preferenze fornite. Controlla il terminale per i dettagli."))
                except Exception as e:
                    rospy.logwarn("Failed publishing suggestions: %s", str(e))

            finally:
                try:
                    self._processing_lock.release()
                except Exception:
                    pass

        except Exception as e:
            rospy.logerr("Unhandled error in cb_chef_request: %s\n%s", str(e), traceback.format_exc())
            self.report_status("exception_fallback")
            suggestions = simple_local_fallback([], max_recipes=self.MAX_RECIPES)
            out_text = self._format_suggestions_text(suggestions, source="fallback:exception")
            try:
                self.pub.publish(String(data=out_text))
                self.pub_robot_speech.publish(String(data="Si è verificato un errore interno: ho applicato una strategia alternativa per le ricette."))
            except Exception:
                pass

    def _format_suggestions_text(self, suggestions, source="fallback"):
        lines = [f"(source={source})"]
        for title, desc in suggestions:
            lines.append(f"--- {title} ---")
            lines.append(desc)
        return "\n".join(lines)

    def _llm_worker(self, ingredients, result_out: dict):
        try:
            system_prompt = ("Sei un assistente disponibile che suggerisce ricette in italiano per uno chef robot."
                             "Data una lista di ingredienti, suggerisci fino a 3 ricette con passaggi semplici e tempi ragionevoli.")
            user_prompt = f"Available ingredients: {', '.join(ingredients)}\nPlease give up to 3 recipe suggestions (title, ingredients list, steps, time)."
            logd("Calling llm_call (legacy)...")
            text = llm_call(system_prompt, user_prompt)
            result_out['text'] = str(text)
            logd("LLM returned length:", len(result_out['text']))
        except Exception as e:
            result_out['error'] = f"{type(e).__name__}: {e}"
            rospy.logerr("Error in llm_call: %s\n%s", str(e), traceback.format_exc())

    def _llm_worker_structured(self, ingredients, dish_type, dietary, result_out: dict):
        try:
            system_prompt = ("Sei un assistente disponibile che suggerisce ricette in italiano per uno chef robot."
                             "Data una lista di ingredienti e delle preferenze/diete, suggerisci fino a 3 ricette con passaggi semplici e tempi ragionevoli.")
            user_prompt = (f"Available ingredients: {', '.join(ingredients)}\n"
                           f"Dish type requested: {dish_type if dish_type else 'any'}\n"
                           f"Dietary restrictions / preferences: {dietary if dietary else 'none'}\n"
                           f"Please give up to {self.MAX_RECIPES} recipe suggestions (title, ingredients list, steps, time)."
                           "\nIMPORTANT: Respect the dish type (prefer recipes of that type) and the dietary restrictions. "
                           "If an ingredient conflicts with dietary restrictions, suggest substitutions or omit it.")
            logd("Calling llm_call (structured)...")
            text = llm_call(system_prompt, user_prompt)
            result_out['text'] = str(text)
            logd("LLM returned length:", len(result_out['text']))
        except Exception as e:
            result_out['error'] = f"{type(e).__name__}: {e}"
            rospy.logerr("Error in llm_call (structured): %s\n%s", str(e), traceback.format_exc())

def main():
    node = SemanticChefNode()
    rospy.spin()

if __name__ == "__main__":
    main()