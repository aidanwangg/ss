# Deploying to Hugging Face Spaces (free)

This gets the app a public URL like `https://<you>-sudoku-solver.hf.space`.
The app runs in Docker; the model is **trained during the build**, so there's
nothing to upload manually.

## One-time setup

1. Create a free account at https://huggingface.co (no credit card).
2. Click **New → Space**.
   - **Owner:** your username.
   - **Space name:** e.g. `sudoku-solver`.
   - **SDK:** choose **Docker** → **Blank**.
   - **Hardware:** the free **CPU basic** tier is enough.
   - Visibility: Public (or Private).
   - Create the Space.

## Push the code

A Space is its own git repo. Add it as a remote and push this project to it.

```bash
# from the repo root, on the branch you want to deploy (e.g. main)
git remote add space https://huggingface.co/spaces/<your-username>/sudoku-solver
git push space main
```

- When prompted for a password, use a **Hugging Face access token**
  (Settings → Access Tokens → New token, role: *write*), not your account
  password.
- If your default branch is named differently, push it to the Space's `main`:
  `git push space <yourbranch>:main`.

That's it. The Space will show a **Building** status, install dependencies,
train the model (a few minutes the first time), then go **Running**. Your URL
appears at the top of the Space page.

## How it works

- `Dockerfile` installs fonts + OpenCV libs, installs `requirements.txt`,
  runs `python train.py` to produce `models/digit_model.keras`, then serves
  with gunicorn on port **7860**.
- The Hugging Face metadata at the top of `README.md` (`sdk: docker`,
  `app_port: 7860`) tells the Space how to run it.

## Updating the live app

Push again and the Space rebuilds automatically:

```bash
git push space main
```

## Notes

- First load after a rebuild (or after the Space sleeps) can take a few seconds
  while TensorFlow and the model load.
- Build time is dominated by installing TensorFlow and training the model.
  To speed up builds you can lower epochs in the `Dockerfile`
  (`python train.py --epochs 6`).
- Free Spaces may sleep when idle and wake on the next visit.
