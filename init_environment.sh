python3 -m venv llm_venv
source llm_venv/bin/activate
pip install ipykernel
python3 -m ipykernel install --user --name=llm_venv --display-name "Python (llm_venv)"
pip install -r requirements.txt

python3 -m spacy download ru_core_news_sm
python3 -m spacy validate

apt update
apt install -y lsof
apt-get install unzip

lsof -i :11434

curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull mistral


unzip ./data/recipes_1.zip -d ./data
unzip ./data/recipes_2.zip -d ./data
unzip ./data/recipes_3.zip -d ./data
