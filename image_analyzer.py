"""
파우더브로우 이미지 분석기 v3
- 일본어 메시지 출력
- 눈썹 오감지 방지 강화
- 선생님 기준 40점 캘리브레이션
- 친절한 톤
"""

import cv2
import numpy as np
import json
import os
import time
import tempfile

TEACHER_REF = {
    "profile_80": [1, 28, 40, 35, 14],
    "gradient_range": 40,
}

PATTERN_REFS = {
    "SOFT": {
        "darkness": [121, 128, 136, 129, 116],
        "density":  [26.1, 53.0, 67.4, 67.5, 40.0],
    },
    "NATURAL": {
        "darkness": [120, 122, 123, 121, 122],
        "density":  [25.1, 54.0, 57.4, 50.6, 35.2],
    },
    "MIX": {
        "darkness": [125, 127, 129, 128, 126],
        "density":  [32.1, 63.7, 62.8, 53.3, 39.9],
    },
}


def detect_brows(image):
    """눈썹 영역 감지 — 가장 큰 1개만 반환, 오감지 철저 방지"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    h, w = gray.shape

    bg_value = np.median(gray)
    threshold = bg_value * 0.72
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 너무 작거나 너무 큰 것 제외
        if area < img_area * 0.01 or area > img_area * 0.30:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0

        # 눈썹 형태: 가로 2.5~6배
        if aspect < 2.5 or aspect > 6.0:
            continue

        # 이미지 상단 20% 제외 (커팅매트/자)
        if y < h * 0.20:
            continue

        # 이미지 하단 20% 제외 (텍스트/이름)
        if (y + bh) > h * 0.80:
            continue

        # 높이가 이미지의 5% 미만 제외
        if bh < h * 0.05:
            continue

        # 초록색 영역 제외 (커팅매트)
        roi_hsv = hsv[y:y+bh, x:x+bw]
        mask_roi = np.zeros((bh, bw), dtype=np.uint8)
        shifted = cnt.copy()
        shifted[:, 0, 0] -= x
        shifted[:, 0, 1] -= y
        cv2.drawContours(mask_roi, [shifted], -1, 255, -1)
        hue_pixels = roi_hsv[:, :, 0][mask_roi > 0]
        if len(hue_pixels) > 0:
            green_ratio = np.sum((hue_pixels > 35) & (hue_pixels < 85)) / len(hue_pixels)
            if green_ratio > 0.3:
                continue

        candidates.append({"contour": cnt, "bbox": (x, y, bw, bh), "area": area})

    if not candidates:
        return []

    # 면적이 가장 큰 1개만 반환
    candidates.sort(key=lambda b: b["area"], reverse=True)
    return [candidates[0]]


def analyze_zones(gray, brow, num_zones=5):
    """5구역 분할 분석"""
    x, y, w, h = brow["bbox"]
    zone_width = w // num_zones
    bg_value = float(np.median(gray))

    zones = []
    for i in range(num_zones):
        zx = x + i * zone_width
        zw = zone_width if i < num_zones - 1 else (w - i * zone_width)

        roi = gray[y:y+h, zx:zx+zw]
        mask_roi = np.zeros_like(roi)
        shifted = brow["contour"].copy()
        shifted[:, 0, 0] -= zx
        shifted[:, 0, 1] -= y
        cv2.drawContours(mask_roi, [shifted], -1, 255, -1)

        brow_pixels = roi[mask_roi > 0]

        if len(brow_pixels) == 0:
            zones.append({"darkness": 0, "density": 0, "pixel_count": 0})
            continue

        darkness = max(0, float(bg_value - np.mean(brow_pixels)))

        dark_threshold = bg_value * 0.85
        dark_count = np.sum(brow_pixels < dark_threshold)
        density = (dark_count / len(brow_pixels)) * 100

        zones.append({
            "darkness": round(darkness, 1),
            "density": round(density, 1),
            "pixel_count": int(len(brow_pixels)),
        })

    return zones


def normalize_to_80(zones):
    max_dark = max(z["darkness"] for z in zones) if zones else 1
    if max_dark == 0:
        return [0] * len(zones)
    scale = 80 / max_dark
    return [round(z["darkness"] * scale, 1) for z in zones]


def generate_feedback(zones, pattern="SOFT"):
    """피드백 생성 — 일본어, 친절한 톤, 40점 캘리브레이션"""
    ref = PATTERN_REFS.get(pattern, PATTERN_REFS["SOFT"])
    ref_dark = ref["darkness"]
    ref_dens = ref["density"]

    improvements = []  # (icon, text)
    total_deduction = 0

    stu_dark = [z["darkness"] for z in zones]
    stu_dens = [z["density"] for z in zones]
    profile_80 = normalize_to_80(zones)
    teacher_profile = TEACHER_REF["profile_80"]

    avg_dark_80 = sum(profile_80) / 5
    stu_avg = sum(stu_dark) / 5
    ref_avg = sum(ref_dark) / 5
    diff_avg = stu_avg - ref_avg
    grad_range = max(profile_80) - min(profile_80)

    # === 1. 레이어링 부족 ===
    if avg_dark_80 < 35:
        improvements.append(("🔴", "全体的にレイヤリング回数が大幅に不足しています。先生は5回以上重ねていますが、この仕上がりは1〜2回程度です。同じ弱い力でもう3回以上重ねてください。重ねれば重ねるほど「面」になります。"))
        total_deduction += 15
    elif avg_dark_80 < 55:
        improvements.append(("🔴", "全体的にレイヤリング回数が足りません。先生は5回以上重ねていますが、この仕上がりは2回程度です。同じ弱い力でもう3回以上重ねてください。重ねれば重ねるほど面になります。"))
        total_deduction += 10

    # === 2. 전체 과다 ===
    if diff_avg > 15:
        improvements.append(("🔴", f"全体的に濃すぎます。力を入れるのではなく、赤ちゃんのほっぺを触るくらいやさしく、同じ弱い力で何回も重ねてください。バウンド高さ3〜5mmを意識しましょう。"))
        total_deduction += 10
    elif diff_avg > 8:
        improvements.append(("🟡", "少し濃い傾向です。圧力をもう少し抑えてみてください。"))
        total_deduction += 5

    # === 3. 오렌지존 분리 ===
    ref_jump_2 = ref_dark[2] - ref_dark[1]
    ref_jump_4 = ref_dark[2] - ref_dark[3]
    stu_jump_2 = stu_dark[2] - stu_dark[1]
    stu_jump_4 = stu_dark[2] - stu_dark[3]

    if stu_jump_2 > ref_jump_2 * 2 + 5 or stu_jump_4 > ref_jump_4 * 2 + 5:
        improvements.append(("🔴", "オレンジゾーンの陰影と外側のグラデーションが分離して見えます。オレンジゾーンから上/横に力を抜きながらレイヤリングをつなげて、2〜3mmのブレンディング区間を作ってください。"))
        total_deduction += 12

    # === 4. 점 간격 과밀 ===
    stu_avg_dens = sum(stu_dens) / 5
    ref_avg_dens = sum(ref_dens) / 5
    dens_diff = stu_avg_dens - ref_avg_dens

    if dens_diff > 20:
        improvements.append(("🔴", "点の間隔が詰まりすぎてグラデーション感がなくなっています。特に上段は点の間隔を広げて、皮膚が見えるように「解いて」あげてください。"))
        total_deduction += 10
    elif dens_diff > 10:
        improvements.append(("🟡", "全体的に密度が高めです。点の間隔をもう少し広げてみてください。"))
        total_deduction += 5

    # === 5. 얼룩 ===
    stu_range = max(stu_dark) - min(stu_dark)
    ref_range_val = max(ref_dark) - min(ref_dark)

    if diff_avg > 10 and stu_range > ref_range_val * 1.5:
        improvements.append(("🟡", "ムラ（まだら）が見えます。均一な圧で丁寧に重ねましょう。均一でないと定着後にまだらに残ってしまいます。"))
        total_deduction += 5

    # === 6. 앞머리 그라데이션 ===
    stu_gradient = stu_dark[2] - stu_dark[0]
    ref_gradient = ref_dark[2] - ref_dark[0]

    if pattern in ("SOFT", "MIX"):
        if stu_gradient < 3:
            improvements.append(("🔴", "眉頭〜中央のグラデーションがほぼありません。眉頭は「消えるように」、中央（オレンジゾーン）は何回も重ねて濃くして、明暗の差を作ってください。"))
            total_deduction += 10
        elif stu_gradient < ref_gradient * 0.5:
            improvements.append(("🟡", "眉頭〜中央のグラデーションがもう少し必要です。眉頭をもっと薄く、中央をもっと濃くして差を作りましょう。"))
            total_deduction += 5

    # === 7. 미두 뭉침 ===
    if profile_80[0] > 8:
        improvements.append(("🟡", f"眉頭がやや濃いです（先生の{profile_80[0]:.0f}倍）。もう少し力を抜いてタッチしてください。眉頭は「空ける」のではなく「弱く何回も重ねて自然に見せる」のが正解です。"))
        total_deduction += 5

    # === 8. 프로파일 평탄 ===
    if grad_range < 15:
        improvements.append(("🔴", f"全体が均一で、グラデーションになっていません。先生の眉は眉頭（薄い）→眉上（濃い）→眉尾（薄い）の曲線ですが、この作品は平坦です。眉頭をもっと薄く、眉上をもっと濃くして、明暗の差を作ってください。"))
        total_deduction += 10
    elif grad_range < 25:
        improvements.append(("🟡", "グラデーションの幅がもう少し欲しいです。オレンジゾーンをもう少し濃く重ねて、眉頭との差を広げてみてください。"))
        total_deduction += 5

    # === 9. 꼬리 역전 ===
    if stu_dark[4] > stu_dark[2]:
        improvements.append(("🟡", "眉尻が中央より濃くなっています。眉尻は自然にフェードアウトするように点を減らし間隔を広げてください。"))
        total_deduction += 5

    score = max(0, min(100, 100 - total_deduction))

    return {
        "score": score,
        "improvements": improvements,
        "profile_80": profile_80,
        "grad_range": round(grad_range, 1),
        "avg_dark_80": round(avg_dark_80, 1),
    }


def detect_pattern(zones):
    profile_80 = normalize_to_80(zones)
    avg_dark = sum(profile_80) / 5

    if avg_dark < 40:
        return "判別不可"

    scores = {}
    for name, ref in PATTERN_REFS.items():
        diff = sum(abs(zones[i]["darkness"] - ref["darkness"][i]) for i in range(5))
        dens_diff = sum(abs(zones[i]["density"] - ref["density"][i]) for i in range(5))
        scores[name] = diff + dens_diff * 0.5

    best = min(scores, key=scores.get)
    second = sorted(scores.values())[1]
    gap = second - scores[best]

    if gap < 10:
        return "判別不可"
    return best


def analyze_image(image_path):
    """메인 분석 — 파일 경로"""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "画像を読み取れません。もう一度送ってください🙏"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brows = detect_brows(image)

    if not brows:
        return {"error": "眉毛を検出できませんでした。眉毛がはっきり見える写真でもう一度送ってください🙏"}

    # 가장 큰 눈썹 1개만 분석 (복수 눈썹은 3패턴 과제일 때만)
    brow = brows[0]
    zones = analyze_zones(gray, brow)
    pattern = detect_pattern(zones)
    target = pattern if pattern in PATTERN_REFS else "SOFT"
    analysis = generate_feedback(zones, target)

    # 프로파일 데이터
    profile_80 = analysis["profile_80"]
    teacher = TEACHER_REF["profile_80"]

    # 구역별 판정
    zone_labels = ["眉頭(前)", "眉頭〜眉上", "眉上(中央)", "眉上〜眉尾", "眉尾(尻)"]
    zone_results = []
    for i in range(5):
        diff = profile_80[i] - teacher[i]
        abs_diff = abs(diff)
        if abs_diff <= 5:
            icon = "✅"
        elif abs_diff <= 10:
            icon = "🟡"
        else:
            icon = "🔴"
        zone_results.append(f"{icon} {zone_labels[i]}: {profile_80[i]:.0f} (先生{teacher[i]}, 差{diff:+.0f})")

    # 가장 진한 구역
    peak_idx = profile_80.index(max(profile_80))
    peak_name = zone_labels[peak_idx]

    return {
        "score": analysis["score"],
        "avg_dark_80": analysis["avg_dark_80"],
        "grad_range": analysis["grad_range"],
        "peak_zone": peak_name,
        "profile_80": profile_80,
        "zone_results": zone_results,
        "improvements": analysis["improvements"],
        "pattern": pattern,
        "brow_count": len(brows),
    }


def analyze_image_bytes(image_bytes):
    """바이트 데이터로 분석"""
    tmp_path = os.path.join(tempfile.gettempdir(), f"brow_{int(time.time() * 1000)}.jpg")
    try:
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        return analyze_image(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def format_line_message(result):
    """LINE 전송용 메시지 — 기존 포맷 유지, 친절한 톤"""
    if "error" in result:
        return result["error"]

    score = result["score"]
    profile = result["profile_80"]
    teacher = TEACHER_REF["profile_80"]

    lines = []
    lines.append("ご提出ありがとうございます！🙇")
    lines.append("添削させていただきます。")
    lines.append(f"📊 分析結果")
    lines.append(f"総合スコア: {score}/100")
    lines.append(f"濃さ: 先生=80 → 学生={result['avg_dark_80']:.0f}")
    lines.append(f"グラデーション幅: 先生=40 → 学生={result['grad_range']:.0f}")
    lines.append(f"一番濃いゾーン: {result['peak_zone']}")

    lines.append("【ゾーン別】")
    for zr in result["zone_results"]:
        lines.append(zr)

    lines.append(f"📈 プロファイル:")
    lines.append(f"学生: {' → '.join(str(int(p)) for p in profile)}")
    lines.append(f"先生: {' → '.join(str(p) for p in teacher)}")

    if result["improvements"]:
        lines.append("【改善ポイント】")
        for icon, text in result["improvements"]:
            lines.append(f"{icon} {text}")

    lines.append("引き続き練習を頑張ってください！")
    lines.append("何かご不明な点がございましたら、お気軽にご質問ください☺️")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_image(sys.argv[1])
        print(format_line_message(result))
    else:
        print("Usage: python image_analyzer.py <image_path>")
