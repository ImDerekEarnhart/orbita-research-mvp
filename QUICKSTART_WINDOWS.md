# Orbita Research MVP — Windows quick start

## First installation

1. Extract the ZIP to a normal folder, for example:

   `C:\Users\Dereks\Downloads\orbita_research_mvp_v0_1_0`

2. Open PowerShell inside that extracted folder.

3. Install the local environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

## Start Orbita

The easiest option is to double-click:

```text
LAUNCH_ORBITA.bat
```

Or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch_mvp.ps1
```

A server window will stay open and the browser should open automatically at:

```text
http://127.0.0.1:8010/
```

Interactive API documentation is at:

```text
http://127.0.0.1:8010/docs
```

Do not type a URL by itself into PowerShell. To open it from PowerShell, use:

```powershell
Start-Process "http://127.0.0.1:8010/"
```

## First research run

1. Enter a case name.
2. Leave the research goal blank for Open Discovery, or enter a specific question.
3. Click **Create case**.
4. Choose a CSV, Excel, JSON, JSONL, PDF, DOCX, text, notebook, or ZIP file and click **Upload**.
5. Click **Compile research plan**.
6. Review the JSON plan. You can edit it and click **Save plan edits as new version**.
7. Click **Approve plan**.
8. Click **Run governed discovery**.
9. Click **Open research dossier**.

The current automated discovery route requires at least one parsed table with at least six rows. Supporting documents are preserved and included as source context, but v0.1 does not yet let document prose silently control the statistical plan.

## Stop Orbita

Return to the server PowerShell window and press:

```text
CTRL+C
```

## Files Orbita creates

The extracted folder will contain:

```text
orbita_mvp.db
orbita_workspace\
```

The database contains the persistent belief graph and append-only events. The workspace contains original uploads, normalized tables, discovery ledgers, approved plans, and generated reports. Back up both together.
