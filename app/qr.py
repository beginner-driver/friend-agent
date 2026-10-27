"""사진 속 QR 코드 읽기. 공지 포스터의 신청 링크 같은 QR 내용을 대화 입력창에 넣어 준다.

Python 라이브러리(pyzbar)는 requirements.txt 로 설치되지만, 실제로 QR 을 읽는 zbar 는
운영체제 라이브러리다. Windows 용 pyzbar 에는 zbar 가 들어 있지만, 리눅스 이미지에는 Dockerfile 로 설치해야 한다.
"""
import io

from app.logs import log
from app.ocr import MAX_BYTES


class QrError(Exception):
    pass


def read_codes(data: bytes) -> list[str]:
    from PIL import Image, UnidentifiedImageError

    if len(data) > MAX_BYTES:
        raise QrError("사진이 5MB보다 큽니다.")
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as error:
        raise QrError("사진 파일을 읽을 수 없습니다.") from error
    try:
        # zbar 라이브러리가 없으면 import 할 때 ImportError 가 난다.
        from pyzbar.pyzbar import decode
    except ImportError as error:
        log("ERROR", "zbar 라이브러리를 찾을 수 없습니다. 실행 환경에 zbar 가 설치되어 있지 않습니다.")
        raise QrError("QR 코드를 읽는 프로그램이 서버에 없습니다.") from error
    codes = [c.data.decode("utf-8", errors="replace") for c in decode(image)]
    log("INFO", "QR 코드 읽기", count=len(codes), chars=sum(len(c) for c in codes))
    return codes
