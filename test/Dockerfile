FROM kubeflownotebookswg/jupyter-pytorch-full:v1.7.0-rc.0
USER root
RUN apt-get update && apt-get install -y openssh-server
RUN echo 'root:root' | chpasswd
RUN echo 'jovyan:root' | chpasswd
RUN sed -i'' -e's/^#PermitRootLogin prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PasswordAuthentication yes$/PasswordAuthentication yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PermitEmptyPasswords no$/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
RUN mkdir -p /run/sshd
EXPOSE 22
CMD ["sh","-c", "/usr/sbin/sshd && jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
 # CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]

# FROM jupyter/base-notebook
# USER root
# RUN apt-get update && apt-get install -y openssh-server
# RUN mkdir -p /run/sshd && /usr/sbin/sshd
# EXPOSE 22
# EXPOSE 8888
# ENV NB_PREFIX /
# CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]