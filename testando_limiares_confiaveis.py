from ultralytics import YOLO

model = YOLO("yolo11n.pt")
img = "https://ultralytics.com/images/bus.jpg"

for conf_threshold in [0.25, 0.5, 0.75, 0.9]:
    results = model.predict(img, conf=conf_threshold, verbose=False)
    n_objetos = len(results[0].boxes)
    print(f"conf={conf_threshold} → {n_objetos} objetos detectados")



#caso vc seja um entusiasta, curioso ou estudante (que nem eu) talvez vc tenha se deparado com algum erro
#de biblioteca. Os comentários abaixo podem lhe ajudar de alguma forma:
#pip install ultralytics.
#rode esse comando no terminal caso vc ainda não tenha a versão correta da biblioteca instalada na sua IDE.
#Isso evita erros como o "ModeuleNotFoundError: no module named 'ultralytics'".
#Após instalar basta rodar o script normalmente.
