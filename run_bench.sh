#!/bin/bash

# ---- CONFIG ----
export BIP300301_ENFORCER="../bip300301-enforcer"
export BITCOIND="../bitcoin-patched-bins/bitcoind"
export BITCOIN_CLI="../bitcoin-patched-bins/bitcoin-cli"
export ELECTRS="../electrs/target/release/electrs"
export THUNDER_APP="target/release/thunder_app"
export DATADIR="/home/silva/.drivechain"
export ELECTRS_HTTP_PORT=3003

# ---- CHECK BINARIES ----
for BIN in "$BIP300301_ENFORCER" "$BITCOIND" "$BITCOIN_CLI" "$ELECTRS" "$THUNDER_APP"; do
  if [ ! -x "$BIN" ]; then
    echo "Warnung: $BIN ist nicht vorhanden oder nicht ausführbar!"
  fi
done

# ---- KILL FUNCTION ----
function cleanup {
  echo "Stoppe bitcoind und electrs..."
  [[ -n "$BITCOIND_PID" ]] && kill $BITCOIND_PID
  [[ -n "$ELECTRS_PID" ]] && kill $ELECTRS_PID
  wait
}
trap cleanup EXIT

# ---- START BITCOIND ----
echo "Starte bitcoind..."
"$BITCOIND" -regtest -txindex -datadir="$DATADIR" > /tmp/bitcoind.log 2>&1 &
BITCOIND_PID=$!

# Warte, bis bitcoind bereit ist (rpc erreichbar)
echo "Warte auf bitcoind..."
while ! "$BITCOIN_CLI" -regtest -datadir="$DATADIR" getblockchaininfo > /dev/null 2>&1; do
  sleep 1
done

# ---- START ELECTRS ----
echo "Starte electrs..."
"$ELECTRS" --network regtest --http-addr 127.0.0.1:$ELECTRS_HTTP_PORT --daemon-dir "$DATADIR" > /tmp/electrs.log 2>&1 &
ELECTRS_PID=$!

# Warte, bis electrs auf Port lauscht
echo "Warte auf electrs..."
while ! ss -ltn | grep ":$ELECTRS_HTTP_PORT " > /dev/null; do
  sleep 1
done

echo "Starte Integrationstest/Benchmark..."
# ---- TESTAUSFÜHRUNG ----
if [ $# -eq 0 ]; then
  TESTS=(deposit_withdraw_roundtrip unknown_withdrawal initial_block_download)
  for TEST in "${TESTS[@]}"; do
    echo
    echo "========================"
    echo " Starte Test: $TEST"
    echo "========================"
    time cargo run --release --example integration_tests "$TEST"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      echo "FEHLER: Test $TEST fehlgeschlagen! (exit code $EXIT_CODE)"
      exit $EXIT_CODE
    fi
  done
  echo
  echo "Alle Integrationstests abgeschlossen!"
else
  echo "Starte Integrationstest/Benchmark..."
  time cargo run --release --example integration_tests "$@"
fi

# Cleanup übernimmt das Trap


# -------------------------------------------------------------
# Integrationstests Thunder / bip300301_enforcer: Übersicht & Aufruf
#
# 1. deposit_withdraw_roundtrip
#    Testziel: Testet die komplette Ein- und Auszahlung (Deposit/Withdrawal)
#    auf der Sidechain inklusive aller relevanten Interaktionen mit den Nodes.
#    Es wird geprüft, ob Einzahlungen korrekt ankommen und Auszahlungen wieder
#    in die Mainchain zurückgeführt werden können.
#    Startbefehl:
#      ./run_bench.sh deposit_withdraw_roundtrip
#
# 2. unknown_withdrawal
#    Testziel: Überprüft die Reaktion des Systems auf unbekannte oder nicht
#    existierende Withdrawal-Anfragen. Es soll sichergestellt werden, dass keine
#    Gelder verloren gehen oder fälschlicherweise transferiert werden, wenn ein
#    Withdrawal nicht erkannt wird.
#    Startbefehl:
#      ./run_bench.sh unknown_withdrawal
#
# 3. initial_block_download
#    Testziel: Testet den Initial Block Download (IBD) Prozess und die
#    Synchronisation zwischen zwei Thunder-Nodes. Es wird geprüft, ob sich ein
#    neuer Node korrekt mit einem existierenden Node synchronisieren kann.
#    Startbefehl:
#      ./run_bench.sh initial_block_download
#
# Allgemeiner Aufruf für alle Tests:
#   ./run_bench.sh

# Weitere Hinweise:
# - Optional kannst du an den Aufruf noch weitere Argumente hängen, z.B.
#   "-- --test-threads 1", falls du single-threaded testen willst.
# - Beispiel für unknown_withdrawal mit einem Thread:
#      ./run_bench.sh unknown_withdrawal -- --test-threads 1
#
# - Die Tests starten immer ihre eigenen Instanzen von bitcoind, electrs usw.
# - Beachte Port-Kollisionen, wenn Tests parallel laufen.
# -------------------------------------------------------------
