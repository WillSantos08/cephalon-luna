import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config.settings        import load_config
from parser.markdown_parser import MarkdownParser
from core.tokenizer         import LunaTokenizer
from core.model             import LunaModel
from core.dataset           import build_loaders
from core.trainer           import Trainer


def main():
    print("""
╔══════════════════════════════════════════╗
     🌙  Cephalon Luna - Treinamento       
            Treino → Teste → Repita         
╚══════════════════════════════════════════╝
    """)

    cfg = load_config("config.yaml")
    cfg.ensure_dirs()

    if "--resume" in sys.argv:
        cfg.training.resume = True

    torch.manual_seed(cfg.hardware.seed)
    device = cfg.resolve_device()

    print(f"  🖥️  Device : {device}")
    print(f"  🌱 Seed   : {cfg.hardware.seed}\n")

    # ── 1. Markdowns ──────────────────────────────────
    print("── 1. Lendo Markdowns ───────────────────────")
    parser = MarkdownParser(cfg.paths.raw_data)
    kb     = parser.parse_all()

    if len(kb) == 0:
        print(f"  ❌ Nenhum par encontrado em {cfg.paths.raw_data}")
        sys.exit(1)

    if cfg.data_aug:
        kb = parser.augment(kb)
        print(f"  🔀 Após augmentation : {len(kb)} pares")

    corpus      = kb.to_corpus()
    corpus_path = os.path.join(cfg.paths.processed_data, "corpus.txt")

    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"  💾 Corpus salvo : {corpus_path}")
    print(f"  📝 Tamanho      : {len(corpus):,} chars\n")

    # ── 2. Tokenizer ──────────────────────────────────
    print("── 2. Tokenizer ─────────────────────────────")
    tok_path = cfg.paths.tokenizer

    if os.path.exists(tok_path) and cfg.training.resume:
        tokenizer = LunaTokenizer.load(tok_path)
    else:
        tokenizer = LunaTokenizer()
        tokenizer.train(corpus, max_vocab=cfg.model.vocab_size)
        tokenizer.save(tok_path)
    print()

    # ── 3. Tokenizar Corpus ───────────────────────────
    print("── 3. Tokenizando Corpus ────────────────────")
    token_ids = tokenizer.encode(corpus)
    print(f"  🔢 Tokens totais : {len(token_ids):,}\n")

    # ── 4. DataLoaders ────────────────────────────────
    print("── 4. DataLoaders ───────────────────────────")
    train_dl, val_dl = build_loaders(
        token_ids,
        context_len = cfg.model.context_len,
        batch_size  = cfg.training.batch_size,
        val_ratio   = cfg.val_ratio,
        num_workers = cfg.hardware.num_workers,
        pin_memory  = cfg.hardware.pin_memory,
    )
    print()

    # ── 5. Modelo ─────────────────────────────────────
    print("── 5. Modelo ────────────────────────────────")
    cfg.model.vocab_size = tokenizer.vocab_size
    best_path = cfg.paths.latest_model

    if cfg.training.resume and os.path.exists(best_path):
        model = LunaModel.load(best_path, device=str(device))
        print(f"  ♻️  Retomando de : {best_path}")
    else:
        model = LunaModel(cfg.model)

    print(
        f"  📐 Arquitetura : "
        f"{cfg.model.num_layers}L × "
        f"{cfg.model.num_heads}H × "
        f"{cfg.model.d_model}D"
    )
    print(f"  📊 Parâmetros  : {model.num_params():,}\n")

    # ── 6. Treino → Teste → Repita ────────────────────
    print("── 6. Treino → Teste → Repita ───────────────")
    trainer = Trainer(
        model     = model,
        tokenizer = tokenizer,
        train_dl  = train_dl,
        val_dl    = val_dl,
        cfg       = cfg,
        device    = device,
    )

    trainer.fit(
        epochs       = cfg.training.epochs,
        test_prompts = cfg.logging.sample_prompts,
    )

    print("  ✅ Treinamento concluído!")
    print(f"  💾 Modelo  → {cfg.paths.latest_model}")
    print(f"  📊 Logs    → {cfg.paths.logs}")
    print("\n  👉 Próximos passos:")
    print("     python cephalon_luna/test.py")
    print("     python cephalon_luna/evaluate.py")


if __name__ == "__main__":
    main()