# Deploying the GUI to Streamlit Community Cloud

Puts the simulator online at a `https://<name>.streamlit.app` URL — anyone you
share the link with can run experiments from a browser, nothing to install.
Free of charge.

## What's already prepared in this repo

- `streamlit_app.py` — the entry point Streamlit Cloud runs (it imports the
  packaged app from `src/tde_lab/app.py`).
- `requirements.txt` — tells the cloud builder to `pip install .[gui]`
  (the package itself plus Streamlit).

## Steps

1. **Push the repository to GitHub** (private works too):

   ```bash
   cd tde_lab
   gh auth login                      # once
   gh repo create tde-lab --private --source . --push
   # or --public when you are ready
   ```

2. **Sign in at [share.streamlit.io](https://share.streamlit.io)** with the
   GitHub account that owns the repo and authorize the Streamlit app.

3. **Create the app**: *New app* →
   - Repository: `<your-user>/tde-lab`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - *Advanced settings* → Python version: **3.12** (anything ≥ 3.10 works)
   - *Deploy*.

   First build takes a few minutes (it installs numpy/scipy/matplotlib).

4. **Share the URL.** For a public repo the app is public. For a private
   repo, open the app's *Settings → Sharing* and add your professor's email
   to the viewer list — they sign in with that address and see the app;
   nobody else can.

## Updating

Just `git push` — Streamlit Cloud redeploys automatically on every push to
the configured branch.

## Limits to be aware of

- Free tier ≈ 1 GB RAM, shared CPU: perfect for **single comparisons and the
  `--quick`-scale sweeps** the GUI exposes; 10k-realization sweeps belong on
  a real machine via the CLI (`tde sweep-sas … --resume`).
- Apps go to sleep after ~12 h without visitors; the first visitor wakes one
  up in ~30 s (harmless, just looks like a slow load).
- Uploaded WAV files are processed in memory and not persisted; nothing is
  stored server-side between sessions.

## If you publish the repository

The git history contains code removed before publication (the fitSignal
port, see MIGRATION.md).  If you prefer the *history* to be clean too,
publish a squashed branch instead of `main`:

```bash
git checkout --orphan public && git commit -m "tde-lab v1.1.0"
git push origin public
# then point Streamlit Cloud (and collaborators) at the 'public' branch
```
