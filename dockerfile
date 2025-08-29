FROM python

WORKDIR /app

COPY *.txt /app
RUN pip install -r requirements.txt

COPY . .
CMD ["jupyter", "notebook", "mini_project.ipynb"]