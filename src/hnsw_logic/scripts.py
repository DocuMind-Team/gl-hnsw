from __future__ import annotations


def build_embeddings_main() -> None:
    from .services.bootstrap import build_app

    build_app().pipeline.build_embeddings()


def build_hnsw_main() -> None:
    from .services.bootstrap import build_app

    build_app().pipeline.build_hnsw()


def profile_docs_main() -> None:
    from .services.bootstrap import build_app

    build_app().pipeline.profile_docs()


def discover_edges_main() -> None:
    from .services.bootstrap import build_app

    build_app().pipeline.discover_edges()


def revalidate_edges_main() -> None:
    from .services.bootstrap import build_app

    build_app().pipeline.revalidate_edges()


def evaluate_main() -> None:
    from .services.bootstrap import build_app

    report = build_app().evaluation.evaluate()
    print(report.model_dump_json(indent=2))


def serve_main() -> None:
    import uvicorn

    uvicorn.run("hnsw_logic.api.app:create_app", factory=True, host="127.0.0.1", port=8000, reload=False)
