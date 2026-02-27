#!/usr/bin/env python3
# dispensa_inspector.py (versione integrata con voice_terminal)
"""
Nodo di servizio per "ispezionare la dispensa" in simulazione.

Comportamento:
- Espone il servizio /dispensa_inspect (std_srvs/Empty).
- Opzionalmente chiama un servizio di motion per la testa (configurabile).
- Chiede all'operatore tramite /robot_speech (che il voice_terminal visualizza/legge)
  e aspetta la risposta su /voice_commands.
- Se non arriva risposta via /voice_commands entro timeout, prova a leggere da stdin (stesso comportamento precedente).
- Pubblica la lista su /detected_ingredients (String, csv) — ora pubblicata con ritardo dopo aver inviato il JSON strutturato.
- Notifica il modulo chef chiamando il servizio /chef_suggestion_service (se disponibile) o pubblicando su /chef_request (JSON).
- Può essere eseguito in modalità async (thread) per non bloccare chi chiama il servizio.
"""
import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import String
import time
import threading
import traceback
import os
import sys
import re
import json

# Configurazione topici/servizi (coerente con il tuo codice)
SERVICE_NAME = "/dispensa_inspect"
DETECTED_TOPIC = "/detected_ingredients"
CHEF_REQUEST_TOPIC = "/chef_request"
CHEF_SERVICE = "/chef_suggestion_service"
HEAD_MOTION_SERVICE_PARAM = "~head_motion_service"
# Topic per sopprimere l'interpretazione locale dei nodi vocali
SUPPRESS_TOPIC = "/suppress_local_nl"

# Params (configurabili via rosparam o env)
DEFAULT_CONFIRM_TIMEOUT = 40.0
CONFIRM_TIMEOUT = float(os.getenv("DISPENSA_CONFIRM_TIMEOUT",
                                  rospy.get_param("~confirm_timeout", DEFAULT_CONFIRM_TIMEOUT)))
ASYNC_MODE = rospy.get_param("~async", True)  # default async: non blocca il chiamante
VOICE_TOPIC = "/voice_commands"
ROBOT_SPEECH = "/robot_speech"
# Ritardo (in secondi) prima di pubblicare il CSV su /detected_ingredients
CSV_PUBLISH_DELAY = float(os.getenv("DISPENSA_CSV_DELAY", "2.5"))

class DispensaInspector:
    def __init__(self):
        rospy.init_node("dispensa_inspector", anonymous=False)
        self.pub = rospy.Publisher(DETECTED_TOPIC, String, queue_size=1)
        self.pub_chef_req = rospy.Publisher(CHEF_REQUEST_TOPIC, String, queue_size=1)
        self.pub_suppress = rospy.Publisher(SUPPRESS_TOPIC, String, queue_size=1)
        self.head_motion_service = rospy.get_param(HEAD_MOTION_SERVICE_PARAM, "")
        self.service = rospy.Service(SERVICE_NAME, Empty, self.handle_inspect)
        rospy.loginfo("dispensa_inspector ready: service=%s head_motion_service='%s' (async=%s) confirm_timeout=%.1f csv_delay=%.2f",
                      SERVICE_NAME, self.head_motion_service, str(ASYNC_MODE), CONFIRM_TIMEOUT, CSV_PUBLISH_DELAY)

        # for listening to voice commands when waiting for reply
        # we won't subscribe continuously (we'll use wait_for_message when needed)
        # but keep publisher for robot_speech
        self.pub_speech = rospy.Publisher(ROBOT_SPEECH, String, queue_size=1)
        rospy.sleep(0.05)  # allow publishers to register

    def try_call_head_motion(self):
        """
        Se è configurato un servizio di motion per la testa, prova a chiamarlo (tipo Empty).
        Non è obbligatorio: se fallisce, logga e procede.
        """
        if not self.head_motion_service:
            rospy.loginfo("No head_motion_service configured; skipping head movement.")
            return False
        try:
            rospy.loginfo("Waiting for head motion service '%s'...", self.head_motion_service)
            rospy.wait_for_service(self.head_motion_service, timeout=5.0)
            srv = rospy.ServiceProxy(self.head_motion_service, Empty)
            srv()
            rospy.loginfo("Head motion service '%s' called successfully.", self.head_motion_service)
            return True
        except rospy.ROSException:
            rospy.logwarn("Head motion service '%s' not available (timeout). Skipping.", self.head_motion_service)
        except Exception as e:
            rospy.logwarn("Error calling head motion service '%s': %s", self.head_motion_service, str(e))
        return False

    def publish_robot_speech(self, text):
        """Publish a prompt/notification on /robot_speech (voice_terminal will display/read it)."""
        try:
            rospy.loginfo("[dispensa_inspector -> robot_speech] %s", text)
            self.pub_speech.publish(String(data=text))
        except Exception as e:
            rospy.logwarn("Failed to publish robot_speech: %s", str(e))

    def _suppress_on(self):
        try:
            self.pub_suppress.publish(String(data="on"))
            rospy.loginfo("Published suppression ON to %s", SUPPRESS_TOPIC)
        except Exception as e:
            rospy.logwarn("Failed to publish suppression ON: %s", str(e))

    def _suppress_off(self):
        try:
            self.pub_suppress.publish(String(data="off"))
            rospy.loginfo("Published suppression OFF to %s", SUPPRESS_TOPIC)
        except Exception as e:
            rospy.logwarn("Failed to publish suppression OFF: %s", str(e))

    def prompt_ingredients_via_terminal(self):
        """
        Chiede all'operatore, tramite stdin, gli ingredienti rilevati.
        Ritorna lista di stringhe normalizzate (lowercase, stripped).
        """
        try:
            print("")
            print("=== DISPENSA INSPECTOR (stdin fallback) ===")
            print("Inserisci gli ingredienti trovati nella dispensa, separati da virgola (es: pomodoro, farina, latte).")
            print("Se vuoi simulare nessun ingrediente, premi invio senza testo.")
            inp = input("Ingredienti: ")
            if inp is None:
                inp = ""
            items = [it.strip().lower() for it in inp.split(",") if it.strip()]
            return items
        except Exception as e:
            rospy.logwarn("Could not read from stdin (maybe running under roslaunch). Error: %s", str(e))
            return None

    def wait_for_voice_reply(self, timeout=CONFIRM_TIMEOUT):
        """
        Attende un messaggio su /voice_commands e lo ritorna come testo (stringa).
        Usa rospy.wait_for_message in modo bloccante con timeout.
        """
        try:
            rospy.loginfo("Waiting for reply on %s (timeout=%.1f s)...", VOICE_TOPIC, timeout)
            msg = rospy.wait_for_message(VOICE_TOPIC, String, timeout=timeout)
            reply = msg.data.strip() if msg and msg.data else ""
            rospy.loginfo("Received voice reply: %s", reply)
            return reply
        except rospy.ROSException:
            rospy.logwarn("Timeout waiting for reply on %s", VOICE_TOPIC)
            return None
        except Exception as e:
            rospy.logwarn("Error while waiting for voice reply: %s", str(e))
            return None

    def publish_ingredients(self, items):
        """
        Pubblica la lista (csv) su /detected_ingredients.
        """
        if items is None:
            rospy.logwarn("No items to publish.")
            return False
        csv = ",".join(items)
        try:
            self.pub.publish(String(data=csv))
            rospy.loginfo("Published simulated detected ingredients on %s: %s", DETECTED_TOPIC, csv if csv else "<none>")
            return True
        except Exception as e:
            rospy.logwarn("Failed to publish detected ingredients: %s", str(e))
            return False

    def notify_chef(self, items_or_obj):
        """
        Opzioni:
         - se esiste il servizio /chef_suggestion_service lo chiami (Empty)
         - altrimenti pubblica su /chef_request il CSV o JSON (i listener lo useranno)
        items_or_obj: può essere una lista di ingredienti (legacy) o un dict già strutturato
        """
        try:
            rospy.wait_for_service(CHEF_SERVICE, timeout=1.0)
            rospy.loginfo("Calling chef suggestion service: %s", CHEF_SERVICE)
            srv = rospy.ServiceProxy(CHEF_SERVICE, Empty)
            srv()
            rospy.loginfo("Chef suggestion service called.")
            return
        except rospy.ROSException:
            # service non disponibile -> publish on topic
            try:
                if isinstance(items_or_obj, dict):
                    payload = json.dumps(items_or_obj, ensure_ascii=False)
                else:
                    payload = ",".join(items_or_obj) if items_or_obj else ""
                rospy.loginfo("Chef service not available; publishing request on %s: %s", CHEF_REQUEST_TOPIC, payload)
                self.pub_chef_req.publish(String(data=payload))
            except Exception as e:
                rospy.logwarn("Failed to publish chef request: %s", str(e))
        except Exception as e:
            rospy.logwarn("Error calling chef service: %s. Publishing on topic instead.", str(e))
            try:
                if isinstance(items_or_obj, dict):
                    payload = json.dumps(items_or_obj, ensure_ascii=False)
                else:
                    payload = ",".join(items_or_obj) if items_or_obj else ""
                self.pub_chef_req.publish(String(data=payload))
            except Exception:
                pass

    def _delayed_publish_csv(self, items, delay_sec=CSV_PUBLISH_DELAY):
        """
        Pubblica il CSV dopo un breve ritardo in background.
        Questo assicura che il flusso strutturato (JSON) arrivi e venga elaborato prima del CSV legacy.
        """
        def _worker():
            try:
                rospy.loginfo("Delayed CSV publisher sleeping for %.2f s before publishing detected_ingredients.", delay_sec)
                time.sleep(delay_sec)
                self.publish_ingredients(items)
            except Exception as e:
                rospy.logwarn("Delayed CSV publisher failed: %s", str(e))
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def interactive_inspect_sequence(self):
        """
        Sequenza vera e propria: chiamata da handle_inspect (sincrona o in thread).

        Ora chiede anche:
         - tipo di piatto desiderato (primo, secondo, entrambi)
         - allergie/intolleranze/preferenze (libero)
        e pubblica su /chef_request un JSON:
           {"ingredients": [...], "dish_type": "primo|secondo|entrambi|none", "dietary": "stringa libero testo"}

        IMPORTANT: per evitare che il flusso legacy (CSV) venga processato prima dello structured,
        pubblichiamo il JSON strutturato **prima** e poi il CSV con un piccolo ritardo in background.
        """
        # ATTENZIONE: attiviamo soppressione locale così altri nodi vocali non interpretano le risposte
        try:
            self._suppress_on()
            rospy.loginfo("Starting interactive dispensa inspection sequence...")
            # 1) prova la testa (opzionale)
            self.try_call_head_motion()

            # Simula piccolo delay per acquisizione
            rospy.loginfo("Simulating camera acquisition...")
            time.sleep(1.0)

            # 2) chiedi via voice_terminal (robot_speech) e aspetta risposta su /voice_commands
            prompt = ("Sono davanti alla dispensa. Dimmi quali ingredienti ci sono, separati da virgola.\n"
                      "Esempio: 'uova, pasta, formaggio'.")
            self.publish_robot_speech(prompt)

            reply_text = self.wait_for_voice_reply(timeout=CONFIRM_TIMEOUT)

            # 3) se non ricevi reply, fallback a stdin (utile se non usi voice_terminal)
            if not reply_text:
                self.publish_robot_speech("Non ho ricevuto una risposta via terminale. Puoi digitare gli ingredienti qui (stdin)?")
                items = self.prompt_ingredients_via_terminal()
                if items is None:
                    # nessun input: inform user and quit gracefully
                    rospy.logwarn("No ingredients provided via voice_commands or stdin.")
                    self.publish_robot_speech("Non ho ricevuto gli ingredienti. Annullata l'ispezione.")
                    return
            else:
                # normalize reply_text: if contains ':' take text after
                if ":" in reply_text:
                    reply_text = reply_text.split(":", 1)[1].strip()
                items = [it.strip().lower() for it in reply_text.split(",") if it.strip()]

            # 4) chiedi il tipo di piatto
            self.publish_robot_speech("Che tipo di piatto preferisci? (primo, secondo, entrambi)")
            dish_reply = self.wait_for_voice_reply(timeout=CONFIRM_TIMEOUT)
            if not dish_reply:
                # fallback a stdin per la preferenza del tipo di piatto
                try:
                    print("\n[DISPENSA] Nessuna risposta vocale per tipo piatto; inserire da stdin (premi invio per 'none'):")
                    dish_reply = input("Tipo piatto (primo/secondo/entrambi): ").strip() if sys.stdin and not sys.stdin.closed else ""
                except Exception:
                    dish_reply = ""
            dish_type = "none"
            if dish_reply:
                dr = dish_reply.lower()
                if "primo" in dr or re.search(r"\b1\b", dr) or "pasta" in dr:
                    dish_type = "primo"
                elif "secondo" in dr or re.search(r"\b2\b", dr) or "carne" in dr or "pesce" in dr:
                    dish_type = "secondo"
                elif "entrambi" in dr or "sia" in dr or "qualsiasi" in dr:
                    dish_type = "entrambi"
                else:
                    # keep original free text if not matched
                    dish_type = dr.strip()

            # 5) chiedi allergie/intolleranze/preferenze
            self.publish_robot_speech("Hai allergie, intolleranze o preferenze alimentari? (es. no lattosio, vegetariano, niente noci).")
            diet_reply = self.wait_for_voice_reply(timeout=CONFIRM_TIMEOUT)
            if not diet_reply:
                # fallback a stdin per la preferenza alimentare
                try:
                    print("\n[DISPENSA] Nessuna risposta vocale per allergie/intolleranze; inserire da stdin (premi invio per 'none'):")
                    diet_reply = input("Allergie/intolleranze/preferenze: ").strip() if sys.stdin and not sys.stdin.closed else ""
                except Exception:
                    diet_reply = ""
            dietary = diet_reply.strip().lower() if diet_reply else ""

            # Prepare structured request for chef
            request_obj = {
                "ingredients": items,
                "dish_type": dish_type,
                "dietary": dietary
            }

            # Publish structured request (JSON) to chef_request topic FIRST
            try:
                json_payload = json.dumps(request_obj, ensure_ascii=False)
                self.pub_chef_req.publish(String(data=json_payload))
                rospy.loginfo("Published chef_request JSON: %s", json_payload)
            except Exception as e:
                rospy.logwarn("Failed to publish chef_request JSON: %s", str(e))
                # fallback: call old notify_chef using CSV immediately
                self.publish_ingredients(items)

            # Publish short message to user and then schedule delayed CSV publication
            csv = ",".join(items) if items else "<nessuno>"
            self.publish_robot_speech("Ricevuto. Ingredienti registrati: " + csv + ". Procedo con la proposta delle ricette.")

            # schedule delayed CSV publish so legacy listeners still receive it,
            # but after structured request had time to be processed
            try:
                self._delayed_publish_csv(items, delay_sec=CSV_PUBLISH_DELAY)
            except Exception as e:
                rospy.logwarn("Failed to schedule delayed CSV publish: %s", str(e))

            rospy.loginfo("Interactive dispensa inspection finished.")
        except Exception as e:
            rospy.logerr("Exception in interactive_inspect_sequence: %s\n%s", str(e), traceback.format_exc())
            self.publish_robot_speech("Si è verificato un errore durante l'ispezione della dispensa.")
        finally:
            # disattiviamo la soppressione in ogni caso
            try:
                self._suppress_off()
            except Exception:
                pass

    def handle_inspect(self, req):
        """
        Handler del servizio /dispensa_inspect.
        Se ASYNC_MODE=True, lancia la sequenza in un thread e ritorna subito.
        Se ASYNC_MODE=False, esegue la sequenza e ritorna al chiamante al termine.
        """
        rospy.loginfo("dispensa_inspect service invoked.")
        if ASYNC_MODE:
            try:
                t = threading.Thread(target=self.interactive_inspect_sequence, daemon=True)
                t.start()
                rospy.loginfo("Interactive inspection launched in background thread (async mode).")
            except Exception as e:
                rospy.logwarn("Failed to start background inspection thread: %s", str(e))
                # fallback to synchronous execution
                self.interactive_inspect_sequence()
        else:
            # blocking mode
            self.interactive_inspect_sequence()

        # return immediately to caller (EmptyResponse)
        return EmptyResponse()

def main():
    node = DispensaInspector()
    rospy.spin()

if __name__ == "__main__":
    main()