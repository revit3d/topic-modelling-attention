import nltk


def download_nltk_data():
    nltk_resources = [
        'punkt_tab',
        'stopwords',
    ]

    for resource in nltk_resources:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource)


if __name__ == '__main__':
    download_nltk_data()
