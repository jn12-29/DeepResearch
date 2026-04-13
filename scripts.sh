# run code sandbox
sudo docker run -it --rm -p 8081:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609

# kill old vllm
kill -9 -$(lsof -t -i:6006)

# download hle

modelscope download --dataset cais/hle