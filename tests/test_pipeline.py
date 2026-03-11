from __future__ import annotations


def test_full_pipeline_and_evaluation(app_container):
    assert app_container.pipeline.build_embeddings()["docs"] == 24
    app_container.pipeline.build_hnsw()
    assert app_container.pipeline.profile_docs()["briefs"] == 24
    discovery = app_container.pipeline.discover_edges()
    assert discovery["edges"] == 16
    report = app_container.evaluation.evaluate()
    assert report.accepted_edge_count == 16
    assert report.edge_precision == 1.0
    assert report.hybrid.recall_at_5 >= report.baseline.recall_at_5
