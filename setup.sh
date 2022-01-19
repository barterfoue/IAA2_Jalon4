mkdir -p ~/.streamlit/
echo "
[general]
email = 'youremail@domain.com'
" > ~/.streamlit/credentials.toml
echo "
[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml