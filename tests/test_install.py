import nltk

from install import download_nltk_data


def test_download_nltk_data_downloads_required_resources(monkeypatch):
    downloaded = []

    def fake_download(resource):
        downloaded.append(resource)

    monkeypatch.setattr(nltk, "download", fake_download)

    download_nltk_data()

    assert downloaded == ["punkt_tab", "stopwords"]
