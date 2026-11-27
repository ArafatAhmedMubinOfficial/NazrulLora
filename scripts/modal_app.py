"""Modal serverless GPU runner for Nazrul-1B.

Usage:
  pip install modal
  modal setup
  modal run scripts/modal_app.py --prompt "কাজী নজরুলের ভঙ্গিতে জাগরণ নিয়ে একটি কবিতা লেখো।"
"""

from pathlib import Path
import modal

APP_NAME = "nazrul-1b"
REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.50.0",
        "peft>=0.14.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.44.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.25.0",
    )
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(REPO_ROOT / "outputs" / "nazrul-lora"), remote_path="/app/outputs/nazrul-lora")
)

hf_cache = modal.Volume.from_name("nazrul-1b-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="T4",
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=120,
)
class NazrulModel:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        from src.chat import load_nazrul_model
        self.model, self.tok = load_nazrul_model(Path("/app/outputs/nazrul-lora"))

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 240, temperature: float = 0.75) -> str:
        import sys
        sys.path.insert(0, "/app")
        from src.chat import generate as run_gen
        return run_gen(self.model, self.tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature)


@app.local_entrypoint()
def main(
    prompt: str = "কাজী নজরুল ইসলামের ভঙ্গিতে স্বাধীনতা ও তারুণ্য নিয়ে একটি কবিতা লেখো।",
    max_new_tokens: int = 240,
    temperature: float = 0.75,
):
    text = NazrulModel().generate.remote(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    print("\n--- Nazrul-1B ---\n" + text)
