"""
파우더브로우 이미지 분석 모듈
- 학생 사진 → 5구역 분할 → 정답 비교 → 문제 위치 표시 이미지 + 피드백 생성
"""

import cv2
import numpy as np
from io import BytesIO
import base64

# ── 정답 데이터 ──────────────────────────────────────────
REFERENCES = {
    "SOFT": {
        "darkness": [121, 128, 136, 129, 116],
        "density":  [26.1, 53.0, 67.4, 67.5, 40.0],
        "desc": "점 간격 좁게, 점 크기 작게, 매그넘 니들, 중앙~하부 명암 강조"
    },
    "NATURAL": {
        "darkness": [120, 122, 123, 121, 122],
        "density":  [25.1, 54.0, 57.4, 50.6, 35.2],
        "desc": "점 크기 크고 간격 넓음, 포인트 니들만, 피부 투과 느낌"
    },
    "MIX": {
        "darkness": [125, 127, 129, 128, 126],
        "density":  [32.1, 63.7, 62.8, 53.3, 39.9],
        "desc": "SOFT+NATURAL 결합, 매그넘+포인트"
    },
}

ZONE_NAMES_KO = ["미두(앞)", "미두~미상", "미상(중앙)", "미상~미미", "미미(꼬리)"]
ZONE_NAMES_JA = ["眉頭(前)", "眉頭〜眉上", "眉上(中央)", "眉上〜眉尾", "眉尾(尻)"]


def detect_brow_region(gray):
    """눈썹 영역을 자동 감지하여 bounding box 반환"""
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 적응형 이진화 (눈썹이 배경보다 진하므로)
    # THRESH_BINARY_INV: 진한 부분이 흰색(전경)이 됨
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    # 모폴로지 연산으로 노이즈 제거 + 영역 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 폴백: 이미지 중앙 영역 사용
        h, w = gray.shape
        return (int(w * 0.1), int(h * 0.3), int(w * 0.8), int(h * 0.4))
    
    # 가장 큰 윤곽선 = 눈썹
    # 면적 기준으로 상위 윤곽선들을 합쳐서 bounding box
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 전체 이미지 면적의 1% 이상인 윤곽선만
    img_area = gray.shape[0] * gray.shape[1]
    big_contours = [c for c in contours if cv2.contourArea(c) > img_area * 0.005]
    
    if not big_contours:
        big_contours = contours[:3]
    
    # 모든 큰 윤곽선의 합집합 bounding box
    all_points = np.vstack(big_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # 약간의 패딩
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + pad * 2)
    h = min(gray.shape[0] - y, h + pad * 2)
    
    return (x, y, w, h)


def analyze_zones(gray, bbox):
    """
    눈썹 영역을 5구역으로 분할하여 각 구역의 darkness / density 계산
    
    Returns:
        zones: [{"darkness": float, "density": float, "rect": (x,y,w,h)}, ...]
    """
    x, y, w, h = bbox
    brow = gray[y:y+h, x:x+w]
    
    zone_width = w // 5
    zones = []
    
    for i in range(5):
        zx = i * zone_width
        zw = zone_width if i < 4 else (w - 4 * zone_width)  # 마지막 구역은 나머지 포함
        zone_img = brow[:, zx:zx+zw]
        
        # darkness: 255에서 뺀 값 (높을수록 진함)
        # 배경(흰색~밝은색) 제외: 200 이하인 픽셀만
        mask = zone_img < 220
        if mask.sum() > 0:
            dark_pixels = zone_img[mask]
            darkness = float(np.mean(255 - dark_pixels))
            # 0~255 → 정답 스케일(~100~140)로 정규화
            # 정답 데이터 범위: 약 105~140
            darkness = 80 + (darkness / 255) * 80
        else:
            darkness = 80.0  # 거의 비어있음
        
        # density: 전경 픽셀 비율 (%)
        total_pixels = zone_img.size
        fg_pixels = np.sum(zone_img < 200)
        density = (fg_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        zones.append({
            "darkness": round(darkness, 1),
            "density": round(density, 1),
            "rect": (x + zx, y, zw, h),
            "zone_idx": i,
        })
    
    return zones


def find_gradient_breaks(zones):
    """
    인접 구역 간 명암 변화를 분석하여 그라데이션 끊김 지점 찾기
    
    Returns:
        breaks: [{"between": (i, i+1), "diff": float, "severity": str}, ...]
    """
    breaks = []
    for i in range(4):
        diff = abs(zones[i+1]["darkness"] - zones[i]["darkness"])
        if diff > 12:
            severity = "심각" if diff > 20 else "주의"
            breaks.append({
                "between": (i, i + 1),
                "between_names_ko": f"{ZONE_NAMES_KO[i]} → {ZONE_NAMES_KO[i+1]}",
                "between_names_ja": f"{ZONE_NAMES_JA[i]} → {ZONE_NAMES_JA[i+1]}",
                "diff": round(diff, 1),
                "severity": severity,
            })
    return breaks


def compare_with_reference(student_zones, ref_name="SOFT"):
    """
    학생 수치를 정답과 비교하여 구역별 피드백 생성
    
    Returns:
        results: [{zone, diff_dark, diff_dens, verdict, feedback_ko, feedback_ja}, ...]
        summary: {score, overall_ko, overall_ja}
    """
    ref = REFERENCES[ref_name]
    results = []
    total_penalty = 0
    
    for i, sz in enumerate(student_zones):
        rd = ref["darkness"][i]
        rdens = ref["density"][i]
        
        dd = sz["darkness"] - rd
        ddens = sz["density"] - rdens
        
        # 판정
        abs_dd = abs(dd)
        if abs_dd <= 8:
            verdict = "양호"
            verdict_ja = "良好"
            icon = "✅"
        elif abs_dd <= 15:
            verdict = "개선권장"
            verdict_ja = "改善推奨"
            icon = "🟡"
            total_penalty += 8
        else:
            verdict = "수정필요"
            verdict_ja = "修正必要"
            icon = "🔴"
            total_penalty += 20
        
        # 피드백 생성
        fb_ko, fb_ja = _generate_zone_feedback(i, dd, ddens, ref_name)
        
        results.append({
            "zone_idx": i,
            "zone_ko": ZONE_NAMES_KO[i],
            "zone_ja": ZONE_NAMES_JA[i],
            "student_dark": sz["darkness"],
            "ref_dark": rd,
            "diff_dark": round(dd, 1),
            "student_dens": sz["density"],
            "ref_dens": rdens,
            "diff_dens": round(ddens, 1),
            "verdict": verdict,
            "verdict_ja": verdict_ja,
            "icon": icon,
            "feedback_ko": fb_ko,
            "feedback_ja": fb_ja,
        })
    
    score = max(0, 100 - total_penalty)
    
    # 전체 요약
    if score >= 80:
        overall_ko = f"전체적으로 {ref_name} 정답과 잘 맞습니다! 잘하고 있어요 ✨"
        overall_ja = f"全体的に{ref_name}の正解とよく合っています！よくできています ✨"
    elif score >= 50:
        overall_ko = f"기본적인 그라데이션은 잡혀 있어요. 아래 포인트를 개선하면 훨씬 좋아집니다!"
        overall_ja = f"基本的なグラデーションはできています。以下のポイントを改善すればもっと良くなります！"
    else:
        overall_ko = f"몇 가지 중요한 수정이 필요해요. 하나씩 고쳐가면 반드시 좋아집니다! 💪"
        overall_ja = f"いくつか重要な修正が必要です。一つずつ直せば必ず良くなります！💪"
    
    summary = {
        "score": score,
        "ref_name": ref_name,
        "overall_ko": overall_ko,
        "overall_ja": overall_ja,
    }
    
    return results, summary


def _generate_zone_feedback(zone_idx, dark_diff, dens_diff, ref_name):
    """구역별 구체적 피드백 생성 (한국어 + 일본어)"""
    zone_ko = ZONE_NAMES_KO[zone_idx]
    zone_ja = ZONE_NAMES_JA[zone_idx]
    fb_ko = ""
    fb_ja = ""
    
    # 명암 피드백
    if abs(dark_diff) > 8:
        if dark_diff > 0:
            fb_ko = f"{zone_ko}: 정답보다 +{dark_diff:.0f} 진합니다. 압력을 줄이거나 점 간격을 넓혀주세요."
            fb_ja = f"{zone_ja}: 正解より+{dark_diff:.0f}濃いです。圧力を減らすか、点の間隔を広げてください。"
        else:
            fb_ko = f"{zone_ko}: 정답보다 {dark_diff:.0f} 연합니다. 패스를 추가하거나 점을 더 촘촘히 넣어주세요."
            fb_ja = f"{zone_ja}: 正解より{dark_diff:.0f}薄いです。パスを追加するか、点をもっと密に入れてください。"
    
    # 밀도 피드백 추가
    if abs(dens_diff) > 15:
        if dens_diff > 0:
            fb_ko += f"\n  → 밀도 과다 ({dens_diff:+.0f}%). 점 간격을 넓혀 피부가 보이도록."
            fb_ja += f"\n  → 密度過多（{dens_diff:+.0f}%）。点の間隔を広げて肌が見えるように。"
        else:
            fb_ko += f"\n  → 밀도 부족 ({dens_diff:+.0f}%). 빈 공간을 채워주세요."
            fb_ja += f"\n  → 密度不足（{dens_diff:+.0f}%）。空白を埋めてください。"
    
    # 특수 구역 피드백
    if zone_idx == 0:  # 미두
        if dens_diff > 20:
            fb_ko += "\n  💡 앞머리가 너무 뚜렷해요. 점 3~5개만 극히 적게, '사라지듯' 마감!"
            fb_ja += "\n  💡 眉頭がはっきりしすぎです。点3〜5個で極わずかに、「消えるように」仕上げて！"
    elif zone_idx == 2:  # 오렌지존
        if dark_diff < -10:
            fb_ko += "\n  💡 오렌지존이 약해요! 여기가 가장 진해야 합니다. 아랫단부터 정확하게 쌓아주세요."
            fb_ja += "\n  💡 オレンジゾーンが弱いです！ここが一番濃くあるべきです。下段から正確に重ねてください。"
    elif zone_idx == 4:  # 미미
        fb_ko += "\n  💡 꼬리는 흩날리듯 연하게 페이드아웃! 끝으로 갈수록 점 수를 줄여주세요."
        fb_ja += "\n  💡 眉尾は舞い散るように薄くフェードアウト！先に行くほど点の数を減らしてください。"
    
    if not fb_ko:
        fb_ko = f"{zone_ko}: 양호 ✅"
        fb_ja = f"{zone_ja}: 良好 ✅"
    
    return fb_ko, fb_ja


def draw_annotated_image(img, bbox, student_zones, results, gradient_breaks, ref_name):
    """
    원본 이미지에 구역 경계선, 수치, 문제 표시를 그려서 반환
    
    Returns:
        annotated_img: numpy array (BGR)
    """
    annotated = img.copy()
    x, y, w, h = bbox
    
    # 구역 색상: 판정에 따라
    colors = {
        "양호": (0, 200, 80),      # 초록
        "개선권장": (0, 180, 255),  # 주황
        "수정필요": (0, 0, 255),    # 빨강
    }
    
    zone_width = w // 5
    
    for i, r in enumerate(results):
        zx = x + i * zone_width
        zw = zone_width if i < 4 else (w - 4 * zone_width)
        
        color = colors.get(r["verdict"], (200, 200, 200))
        
        # 구역 테두리
        cv2.rectangle(annotated, (zx, y), (zx + zw, y + h), color, 2)
        
        # 구역 번호 + 판정 아이콘
        label = f"Z{i+1}"
        cv2.putText(annotated, label, (zx + 4, y - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 수치 표시 (구역 아래)
        dark_text = f"D:{r['student_dark']:.0f}"
        diff_text = f"({r['diff_dark']:+.0f})"
        dens_text = f"{r['student_dens']:.0f}%"
        
        text_y = y + h + 18
        cv2.putText(annotated, dark_text, (zx + 2, text_y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.putText(annotated, diff_text, (zx + 2, text_y + 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(annotated, dens_text, (zx + 2, text_y + 32),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        # 수정필요 구역에 X 표시
        if r["verdict"] == "수정필요":
            cx = zx + zw // 2
            cy = y + h // 2
            size = min(zw, h) // 4
            cv2.line(annotated, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 3)
            cv2.line(annotated, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), 3)
    
    # 그라데이션 끊김 표시: 구역 경계에 번개 모양 화살표
    for brk in gradient_breaks:
        i = brk["between"][0]
        bx = x + (i + 1) * zone_width
        by_top = y - 5
        by_bot = y + h + 5
        
        # 끊김 위치에 빨간 점선
        for yy in range(y, y + h, 6):
            cv2.line(annotated, (bx, yy), (bx, min(yy + 3, y + h)), (0, 0, 255), 2)
        
        # ⚡ 끊김 라벨
        severity_color = (0, 0, 255) if brk["severity"] == "심각" else (0, 150, 255)
        cv2.putText(annotated, f"BREAK", (bx - 20, y - 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, severity_color, 2)
        cv2.putText(annotated, f"d={brk['diff']:.0f}", (bx - 16, y - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.33, severity_color, 1)
    
    # 상단에 정보 바
    bar_h = 30
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], bar_h), (40, 40, 40), -1)
    score = max(0, 100 - sum(20 if r["verdict"] == "수정필요" else 8 if r["verdict"] == "개선권장" else 0 for r in results))
    cv2.putText(annotated, f"vs {ref_name} | Score: {score}/100", (10, 20),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    return annotated


def generate_line_message(results, summary, gradient_breaks, lang="ja"):
    """LINE으로 보낼 텍스트 메시지 생성"""
    
    if lang == "ja":
        msg = f"ご提出ありがとうございます！🙇\n"
        msg += f"添削させていただきます。\n\n"
        msg += f"📊 分析結果（vs {summary['ref_name']}正解）\n"
        msg += f"総合スコア: {summary['score']}/100点\n"
        msg += f"{summary['overall_ja']}\n\n"
        
        # 구역별 결과
        msg += "【ゾーン別分析】\n"
        for r in results:
            msg += f"{r['icon']} {r['zone_ja']}: 明暗{r['diff_dark']:+.0f} / 密度{r['diff_dens']:+.0f}%\n"
        
        # 그라데이션 끊김
        if gradient_breaks:
            msg += "\n⚡ グラデーション途切れ検出：\n"
            for brk in gradient_breaks:
                msg += f"🔴 {brk['between_names_ja']} 間で明暗差{brk['diff']:.0f}（{brk['severity']}）\n"
                msg += f"  → この境界に2〜3mmのブレンディング区間を作って、力を抜いてやさしくつなげてください！\n"
        
        # 개선 포인트
        bad = [r for r in results if r["verdict"] != "양호"]
        if bad:
            msg += "\n【改善ポイント】\n"
            for r in bad:
                msg += f"{r['feedback_ja']}\n\n"
        
        good = [r for r in results if r["verdict"] == "양호"]
        if good:
            msg += "【よくできている部分】\n"
            for r in good:
                msg += f"✅ {r['zone_ja']}: 正解範囲内です！\n"
        
        msg += "\n引き続き練習を頑張ってください！\n何かご不明な点がございましたら、お気軽にご質問ください☺️"
    
    else:  # ko
        msg = f"제출 감사합니다! 🙇\n"
        msg += f"첨삭해드리겠습니다.\n\n"
        msg += f"📊 분석 결과 (vs {summary['ref_name']} 정답)\n"
        msg += f"종합 점수: {summary['score']}/100점\n"
        msg += f"{summary['overall_ko']}\n\n"
        
        msg += "【구역별 분석】\n"
        for r in results:
            msg += f"{r['icon']} {r['zone_ko']}: 명암{r['diff_dark']:+.0f} / 밀도{r['diff_dens']:+.0f}%\n"
        
        if gradient_breaks:
            msg += "\n⚡ 그라데이션 끊김 감지:\n"
            for brk in gradient_breaks:
                msg += f"🔴 {brk['between_names_ko']} 사이에서 명암차 {brk['diff']:.0f} ({brk['severity']})\n"
                msg += f"  → 이 경계에 2~3mm 블렌딩 구간을 만들어 힘 빼고 살살 이어주세요!\n"
        
        bad = [r for r in results if r["verdict"] != "양호"]
        if bad:
            msg += "\n【개선 포인트】\n"
            for r in bad:
                msg += f"{r['feedback_ko']}\n\n"
        
        good = [r for r in results if r["verdict"] == "양호"]
        if good:
            msg += "【잘 된 부분】\n"
            for r in good:
                msg += f"✅ {r['zone_ko']}: 정답 범위 안에 있어요!\n"
        
        msg += "\n계속 연습 화이팅! 궁금한 점 있으면 언제든 질문해주세요 ☺️"
    
    return msg


def full_analysis(image_bytes, ref_name="SOFT"):
    """
    전체 파이프라인: 이미지 바이트 → 분석 결과 + 표시 이미지 + 메시지
    
    Args:
        image_bytes: 이미지 바이너리 데이터
        ref_name: "SOFT", "NATURAL", "MIX" 중 하나
    
    Returns:
        {
            "annotated_image_bytes": bytes (JPEG),
            "message_ja": str,
            "message_ko": str,
            "score": int,
            "results": list,
            "gradient_breaks": list,
        }
    """
    # 이미지 로드
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "이미지를 읽을 수 없습니다"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 눈썹 영역 감지
    bbox = detect_brow_region(gray)
    
    # 2. 5구역 분석
    student_zones = analyze_zones(gray, bbox)
    
    # 3. 그라데이션 끊김 찾기
    gradient_breaks = find_gradient_breaks(student_zones)
    
    # 4. 정답 비교
    results, summary = compare_with_reference(student_zones, ref_name)
    
    # 5. 표시 이미지 생성
    annotated = draw_annotated_image(img, bbox, student_zones, results, gradient_breaks, ref_name)
    
    # JPEG 인코딩
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_bytes = buf.tobytes()
    
    # 6. 메시지 생성
    message_ja = generate_line_message(results, summary, gradient_breaks, lang="ja")
    message_ko = generate_line_message(results, summary, gradient_breaks, lang="ko")
    
    return {
        "annotated_image_bytes": annotated_bytes,
        "annotated_image_base64": base64.b64encode(annotated_bytes).decode(),
        "message_ja": message_ja,
        "message_ko": message_ko,
        "score": summary["score"],
        "ref_name": ref_name,
        "results": results,
        "gradient_breaks": gradient_breaks,
        "student_zones": student_zones,
    }


def detect_pattern_type(student_zones):
    """
    학생 수치를 기반으로 가장 가까운 패턴 자동 추정
    (학생이 패턴을 명시하지 않은 경우 사용)
    """
    best_match = None
    best_score = float('inf')
    
    for name, ref in REFERENCES.items():
        diff_sum = sum(
            abs(sz["darkness"] - rd) 
            for sz, rd in zip(student_zones, ref["darkness"])
        )
        if diff_sum < best_score:
            best_score = diff_sum
            best_match = name
    
    return best_match
