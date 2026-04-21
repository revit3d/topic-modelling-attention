import pandas as pd
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups

from cartm import ContextTopicModel, AttentiveTopicModel
from cartm.preprocessing import (
    CorpusLoader,
    BatchedCorpusLoader,
    build_bow,
)
from cartm.metrics import (
    PerplexityMetric,
    NPMICoherenceMetric,
    SparsityMetric,
    TopicVarianceMetric,
)


if __name__ == "__main__":
    data = fetch_20newsgroups(data_home='./data/', subset='all').data
    preprocessor = CorpusLoader()
    tokenized_data, document_bounds = preprocessor.fit_transform(data)
    assert preprocessor.vocabulary is not None
    vocab_size = len(preprocessor.vocabulary)
    loader = BatchedCorpusLoader(
        data=tokenized_data,
        doc_bounds=document_bounds,
        batch_size=10_000,
    )
    bow = build_bow(tokenized_data, document_bounds, vocab_size)

    experiment_results = []
    for n_topics in [100, 10]:
        for ctx_len in [1000, 100, 50, 10]:
            for gamma in [0.001, 0.01, 0.1, 0.5, 0.9]:
                for model_type in [ContextTopicModel, AttentiveTopicModel]:
                    config = {
                        "n_topics": n_topics,
                        "ctx_len": ctx_len,
                        "gamma": gamma,
                        "model_type": model_type.__name__,
                    }
                    metrics = [
                        PerplexityMetric(tag='perplexity'),
                        SparsityMetric(tag='sparsity'),
                        TopicVarianceMetric(top_k=10, tag='topic variance'),
                        NPMICoherenceMetric(bow=bow, top_k=10, tag='coherence'),
                    ]
                    model = model_type(
                        vocab_size=vocab_size,
                        ctx_len=ctx_len,
                        n_topics=n_topics,
                        gamma=gamma,
                        metrics=metrics,
                    )
                    model.fit(
                        data=loader,
                        max_iter=50,
                        tol=0.0,
                        verbose=0,
                        seed=42,
                    )
                    result = {metric.tag: metric.history for metric in metrics}
                    result.update(config)
                    experiment_results.append(result)
                    print('Finished training with config')
                    pprint(config)
    pd.DataFrame(experiment_results).to_csv('./context_experiments_results.csv')
