from langchain.text_splitter import CharacterTextSplitter


def split_text(text: str) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)
