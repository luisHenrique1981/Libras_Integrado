"""Demo por janela temporal. Requer mediaPipe API clássica `solutions.holistic`.

IMPORTANTE: DWPose->MediaPipe tem domínio diferente; saída pode ser incorreta.
O programa não utiliza legendas armazenadas para escolher a resposta.
"""
import argparse
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from .representacao import mediapipe_frame, mediapipe_sequencia
from .avaliar import abrir_modelo, traduzir_features


def executar(args):
    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'holistic'):
        raise RuntimeError('Instale uma versão compatível com mp.solutions.holistic; ex.: mediapipe==0.10.21')
    modelo,meta,device=abrir_modelo(args.modelo,args.cpu)
    camera=cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError('Não foi possível abrir a webcam')
    buffer=deque(maxlen=args.janela)
    ultimo_texto='Capturando movimentos...'
    contador=0
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,model_complexity=1,
            min_detection_confidence=0.5,min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ok,frame=camera.read()
                if not ok:break
                frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                result=holistic.process(rgb)
                raw = mediapipe_frame(result)
                # A confianca de landmarks da mao na API classica nao equivale
                # a score de detecao. Usar presenca explicita para a janela.
                mao_visivel = (result.left_hand_landmarks is not None or
                               result.right_hand_landmarks is not None)
                buffer.append((raw, mao_visivel))
                contador += 1
                if len(buffer) < args.janela:
                    ultimo_texto = f'Capturando sequencia: {len(buffer)}/{args.janela}'
                elif contador % args.a_cada == 0:
                    vistos = sum(visivel for _, visivel in buffer)
                    if vistos < args.janela * 0.25:
                        ultimo_texto = 'Sinais insuficientes (maos pouco visiveis)'
                    else:
                        frames = np.stack([p for p, _ in buffer])
                        features = mediapipe_sequencia(frames)
                        ultimo_texto = traduzir_features(features, modelo, meta, device)
                        if not ultimo_texto:
                            ultimo_texto = 'Sem traducao gerada'
                for mao in (result.left_hand_landmarks, result.right_hand_landmarks):
                    if mao is not None:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, mao, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.putText(frame,'Libras -> Portugues (EXPERIMENTAL)',(15,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
                # OpenCV pode não renderizar corretamente caracteres acentuados.
                for n,parte in enumerate([ultimo_texto[i:i+65] for i in range(0,min(len(ultimo_texto),180),65)]):
                    cv2.putText(frame,parte,(15,65+n*26),cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,255,255),1)
                cv2.imshow('Tradutor de Libras - prototipo',frame)
                if cv2.waitKey(1)&0xFF in (27,ord('q')):break
    finally:
        camera.release();cv2.destroyAllWindows()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--modelo',default='modelos/tradutor.pt')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--janela',type=int,default=64)
    p.add_argument('--a-cada',type=int,default=24)
    p.add_argument('--cpu',action='store_true')
    executar(p.parse_args())
