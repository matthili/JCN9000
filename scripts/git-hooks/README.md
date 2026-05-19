# Git-Hooks fuer dieses Repo

Versionierte Git-Hooks. Werden nicht automatisch von `git clone` aktiviert --
jeder Klon muss sie einmal aktivieren.

## Aktivierung (einmalig pro Klon)

**WSL2 oder Git-Bash:**
```bash
git config core.hooksPath scripts/git-hooks
chmod +x scripts/git-hooks/pre-commit
```

Damit zeigt Git statt auf `.git/hooks/` auf diesen Ordner. Alle Hooks hier
werden automatisch ausgefuehrt, wenn die jeweiligen Git-Aktionen passieren.

## Deaktivieren

```bash
git config --unset core.hooksPath
```

## Was die Hooks tun

### `pre-commit`

Laeuft automatisch bei `git commit`, bevor der Commit zustande kommt.

**Was passiert:**
1. Pruefen: sind `.puml`-Dateien im Commit-Staging?
2. Falls ja: PlantUML rufen und PNG neben jeder PUML rendern.
3. Frisch gerenderte PNGs in den laufenden Commit aufnehmen.

**Was verhindert wird:** dass die `.puml`-Quellen geaendert werden, die PNGs
aber nicht aktualisiert mitgehen. Sonst sieht in der Doku jemand ein Bild,
das nicht mehr zum Quelltext passt.

**Voraussetzungen:**
- `java` im PATH
- `plantuml.jar` unter `C:\Tools\PlantUML\plantuml.jar`
  (sowohl WSL2-Pfad `/mnt/c/...` als auch Git-Bash-Pfad `/c/...` werden geprueft)

**Wenn etwas fehlt:** Hook gibt eine Warnung aus, der Commit laeuft trotzdem
durch. Bei einem PUML-Syntaxfehler bricht der Commit ab.

**Manuell testen:**
```bash
./scripts/git-hooks/pre-commit
```
