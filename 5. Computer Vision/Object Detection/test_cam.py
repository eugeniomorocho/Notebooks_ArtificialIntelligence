import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
else:
    print("✅ Cámara detectada. Presiona ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se puede leer el frame.")
            break
        cv2.imshow("Cámara de prueba", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()