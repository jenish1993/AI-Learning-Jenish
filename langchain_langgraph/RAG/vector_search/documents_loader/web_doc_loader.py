import requests
import bs4
from langchain_core.documents import Document


#loads a documents from web app.
def load_web_page(page_url: str, bs_kwargs: dict | None = None) -> list[Document]:
    #create a web request.
    response = requests.get(page_url)
    response.raise_for_status()

    #beautify bs4 response.
    soup = bs4.BeautifulSoup(response.text, 'html.parser', **(bs_kwargs or {}))

    #return response as text.
    return [Document(page_content=soup.get_text(), metadata={"source": page_url})]

# # test it
# bs_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
# doc = load_web_page("https://lilianweng.github.io/posts/2023-06-23-agent/", 
#                     bs_kwargs={
#                         "parse_only": bs_strainer
#                     })

# print(f"Total characters: {len(doc[0].page_content)}")