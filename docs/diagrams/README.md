# Architektur-Diagramme

PlantUML-Quellen fuer die wichtigsten Architektur-Diagramme. Pro `.puml`-Datei
liegt nach dem Rendern ein gleichnamiges `.png` daneben.

## Rendern

PlantUML-CLI installieren (Java erforderlich) und im Repo-Root:

```bash
# Einzeln
plantuml -tpng docs/diagrams/system_overview.puml
plantuml -tpng docs/diagrams/inference_server.puml

# Alle auf einmal
plantuml -tpng docs/diagrams/*.puml
```

Auf Windows alternativ ueber das offizielle JAR:

```powershell
java -jar plantuml.jar -tpng docs\diagrams\*.puml
```

## Inhalt

| Datei | Was es zeigt |
|---|---|
| `system_overview.puml` | Gesamt-Architektur vom Code im NN-Repo bis zur Web-App im Browser. Engine, Spieler-Typen, Datengen, Training, Eval, Release-Pipeline. |
| `inference_server.puml` | Wie der `batched-gpu`-Modus (im Eval und in der MCTS-Datengen) viele parallele Spiele auf eine GPU buendelt. Game-Threads, Queue, Inferenz-Server. |

## Wann ein neues Diagramm dazukommt

Wenn eine Architektur-Entscheidung getroffen wird, die ohne Bild schwer zu
erklaeren ist (z.B. spaeter ein AlphaZero-Stil MCTS-bei-Inferenz oder ein
Continuous-Learning-Loop), gehoert sie hier hinein. Pro Diagramm eine `.puml`
mit kurzem Kommentar oben, was es zeigt und wie es gerendert wird.
