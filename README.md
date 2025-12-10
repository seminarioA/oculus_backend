## ¿Como ejecutarlo?
En tu terminal (Windows/macOS/Linux), ejecuta lo siguiente:
```bash
docker run -p 80:8000 docker.io/aleseminario/oculus:latest
```

Con GPU:
```bash
docker run --gpus all -p 80:8000 docker.io/aleseminario/oculus:latest
```
> Nota: Asegurate de tener instalado Docker.

> ¿No funciona?
>
> Revisa que tu Docker Desktop este abierto/ejecutandose.
