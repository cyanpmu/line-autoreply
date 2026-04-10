"""
파우더브로우 이미지 분석기 v2
- v2 강화 채점 (선생님 피드백 반영)
- OpenCV만 사용 (무료)
- analyze_image(path) 또는 analyze_image_bytes(bytes) 호출
"""

import cv2
import numpy as np
import json
import os
import tempfile

# 선생님 기준 데이터
TEACHER_REF = {
    "profile_80": [1, 28, 40, 35, 14],
    "raw_darkness": 61.3,
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

COMMON_PROBLEMS = {
    "layering": "レイヤリング不足",
    "midu_dark": "眉頭が濃すぎ",
    "separation": "オレンジゾーン分離",
    "gap": "区間の途切れ",
    "flat_profile": "プロファイル平坦",
    "no_vertical": "垂直グラデーションなし",
    "pressure_high": "圧/深さ過多",
    "density_high": "点の間隔が詰まりすぎ",
    "blotchy": "ムラ/不均一",
    "no_head_gradient": "眉頭グラデーションなし",
}


def detect_brows(image):
    """눈썹 영역 감지 (복수 눈썹 지원)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    bg_value = np.median(gray)
    threshold = bg_value * 0.75
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    brows = []
    min_area = image.shape[0] * image.shape[1] * 0.005

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 1.5 < aspect < 8.0:
                brows.append({"contour": cnt, "bbox": (x, y, w, h), "area": area})

    brows.sort(key=lambda b: b["bbox"][0])
    return brows


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
        density = (dark_count / len(brow_pixels)) * 100 if len(brow_pixels) > 0 else 0

        zones.append({
            "darkness": round(darkness, 1),
            "density": round(density, 1),
            "pixel_count": int(len(brow_pixels)),
        })

    return zones


def normalize_to_80(zones):
    """선생님 80기준으로 정규화"""
    max_dark = max(z["darkness"] for z in zones) if zones else 1
    if max_dark == 0:
        return [0] * len(zones)
    scale = 80 / max_dark
    return [round(z["darkness"] * scale, 1) for z in zones]


def generate_feedback_v2(zones, pattern="SOFT"):
    """v2 강화 피드백 생성 — 선생님 기준 40점 캘리브레이션"""
    ref = PATTERN_REFS.get(pattern, PATTERN_REFS["SOFT"])
    ref_dark = ref["darkness"]
    ref_dens = ref["density"]

    feedback = []
    problems = []
    total_deduction = 0

    stu_dark = [z["darkness"] for z in zones]
    stu_dens = [z["density"] for z in zones]
    profile_80 = normalize_to_80(zones)

    # === 1. 전체 레이어링 체크 ===
    avg_dark_80 = sum(profile_80) / 5
    if avg_dark_80 < 35:
        feedback.append({"type": "critical", "area": "全体レイヤリング", "weight": 15,
            "msg": f"レイヤリングが大幅に不足しています（80基準平均{avg_dark_80:.0f}）。同じ弱い力で5回以上重ねてください。"})
        problems.append("layering")
        total_deduction += 15
    elif avg_dark_80 < 55:
        feedback.append({"type": "critical", "area": "レイヤリング不足", "weight": 12,
            "msg": f"レイヤリングが不足しています（80基準平均{avg_dark_80:.0f}）。パスを追加してください。"})
        problems.append("layering")
        total_deduction += 12

    # === 2. 전체 과다 (압/깊이) ===
    stu_avg = sum(stu_dark) / 5
    ref_avg = sum(ref_dark) / 5
    diff_avg = stu_avg - ref_avg

    if diff_avg > 15:
        feedback.append({"type": "critical", "area": "圧/深さ過多", "weight": 12,
            "msg": f"全体的に濃すぎます（差+{diff_avg:.0f}）。赤ちゃんのほっぺくらいやさしく！濃くしたい部分は強く押すのではなく、弱い力で何回も重ねます。バウンド高さ3〜5mmを意識。"})
        problems.append("pressure_high")
        total_deduction += 12
    elif diff_avg > 8:
        feedback.append({"type": "warning", "area": "やや濃い", "weight": 6,
            "msg": f"少し濃い傾向です（差+{diff_avg:.0f}）。圧力を少し抑えてください。"})
        total_deduction += 6

    # === 3. 오렌지존 분리 ===
    z3 = stu_dark[2]
    z2 = stu_dark[1]
    z4 = stu_dark[3]
    ref_jump_2 = ref_dark[2] - ref_dark[1]
    ref_jump_4 = ref_dark[2] - ref_dark[3]
    stu_jump_2 = z3 - z2
    stu_jump_4 = z3 - z4

    if stu_jump_2 > ref_jump_2 * 2 + 5 or stu_jump_4 > ref_jump_4 * 2 + 5:
        feedback.append({"type": "critical", "area": "オレンジゾーン分離", "weight": 15,
            "msg": "オレンジゾーン（Z3）が周囲と分離して「帯」のように浮いて見えます。オレンジゾーンから上/横にレイヤリングをつなげて、2〜3mmのブレンディング区間を作ってください。"})
        problems.append("separation")
        total_deduction += 15

    # === 4. 점 간격 과밀 ===
    stu_avg_dens = sum(stu_dens) / 5
    ref_avg_dens = sum(ref_dens) / 5
    dens_diff = stu_avg_dens - ref_avg_dens

    if dens_diff > 15:
        feedback.append({"type": "critical", "area": "点の間隔が詰まりすぎ", "weight": 12,
            "msg": f"点の間隔が狭すぎてグラデーション感がありません（密度{stu_avg_dens:.0f}% vs 正解{ref_avg_dens:.0f}%）。上段は点の間隔を広げて皮膚が見えるように「解いて」ください。"})
        problems.append("density_high")
        total_deduction += 12
    elif dens_diff > 8:
        feedback.append({"type": "warning", "area": "密度やや高い", "weight": 6,
            "msg": f"全体密度が高めです（{stu_avg_dens:.0f}% vs 正解{ref_avg_dens:.0f}%）。"})
        total_deduction += 6

    # === 5. 얼룩/불균일 ===
    stu_range = max(stu_dark) - min(stu_dark)
    ref_range_val = max(ref_dark) - min(ref_dark)

    if diff_avg > 10 and stu_range > ref_range_val * 1.5:
        feedback.append({"type": "critical", "area": "ムラ/不均一", "weight": 10,
            "msg": f"濃い上にムラが目立ちます（明暗幅{stu_range:.0f} vs 正解{ref_range_val:.0f}）。均一な圧で丁寧に重ねてください。均一でないと定着後にまだらに残ります。"})
        problems.append("blotchy")
        total_deduction += 10

    # === 6. 앞머리 그라데이션 ===
    stu_gradient = stu_dark[2] - stu_dark[0]
    ref_gradient = ref_dark[2] - ref_dark[0]

    if pattern in ("SOFT", "MIX"):
        if stu_gradient < 3:
            feedback.append({"type": "critical", "area": "眉頭グラデーションなし", "weight": 15,
                "msg": f"眉頭→オレンジゾーンのグラデーションがほぼありません（明暗差{stu_gradient:.0f}、正解{ref_gradient:.0f}）。眉頭は極薄く（間隔0.7〜1.0mm）、中央は何回も重ねて濃く！"})
            problems.append("no_head_gradient")
            total_deduction += 15
        elif stu_gradient < ref_gradient * 0.4:
            feedback.append({"type": "critical", "area": "グラデーション不足", "weight": 12,
                "msg": f"眉頭→オレンジゾーンのグラデーションが大幅に不足（{stu_gradient:.0f} vs 正解{ref_gradient:.0f}）。"})
            problems.append("no_head_gradient")
            total_deduction += 12
        elif stu_gradient < ref_gradient * 0.6:
            feedback.append({"type": "warning", "area": "グラデーション弱い", "weight": 8,
                "msg": f"グラデーションがやや弱いです（{stu_gradient:.0f} vs 正解{ref_gradient:.0f}）。"})
            total_deduction += 8
        else:
            feedback.append({"type": "good", "area": "水平グラデーション", "weight": 0,
                "msg": "眉頭→オレンジゾーンのグラデーション方向は合っています！"})

    # === 7. Z1→Z2→Z3 연속성 ===
    z1z2 = stu_dark[1] - stu_dark[0]
    z2z3 = stu_dark[2] - stu_dark[1]

    if z1z2 < 2 and z2z3 > 8:
        feedback.append({"type": "critical", "area": "眉頭〜中央の途切れ", "weight": 12,
            "msg": "眉頭と眉頭〜眉上がほぼ同じ濃さなのに、オレンジゾーンで急に濃くなっています。眉頭→中央まで徐々に濃くなるグラデーションが必要です。"})
        problems.append("gap")
        total_deduction += 12

    # === 8. 미두 뭉침 ===
    if profile_80[0] > 8:
        feedback.append({"type": "warning", "area": "眉頭が濃すぎ", "weight": 8,
            "msg": f"眉頭が濃すぎます（80基準{profile_80[0]:.0f}、先生は1〜3）。「消えるように」仕上げてください。点3〜5個だけ、間隔0.7〜1.0mmで極わずかに。"})
        problems.append("midu_dark")
        total_deduction += 8

    if len(stu_dens) > 0 and len(ref_dens) > 0 and stu_dens[0] > ref_dens[0] * 1.4:
        feedback.append({"type": "warning", "area": "眉頭密度過多", "weight": 6,
            "msg": f"眉頭の密度が高すぎます（{stu_dens[0]:.0f}% vs 正解{ref_dens[0]:.0f}%）。点を減らして間隔を広げ、スーッと現れる感じに。"})
        total_deduction += 6

    # === 9. 프로파일 평탄 ===
    grad_range = max(profile_80) - min(profile_80)
    if grad_range < 15:
        feedback.append({"type": "critical", "area": "プロファイル平坦", "weight": 10,
            "msg": f"グラデーション幅が狭すぎます（幅{grad_range:.0f}、先生は40）。オレンジゾーンは濃く重ね、眉頭/眉尻は力を抜いて。"})
        problems.append("flat_profile")
        total_deduction += 10
    elif grad_range < 20:
        feedback.append({"type": "warning", "area": "プロファイルやや平坦", "weight": 6,
            "msg": f"グラデーション幅がやや狭いです（幅{grad_range:.0f}、先生は40）。"})
        problems.append("flat_profile")
        total_deduction += 6

    # === 10. 꼬리 페이드아웃 ===
    tail_drop = stu_dark[3] - stu_dark[4]
    ref_tail_drop = ref_dark[3] - ref_dark[4]

    if tail_drop < 0 and ref_tail_drop > 3:
        feedback.append({"type": "critical", "area": "眉尻逆転", "weight": 10,
            "msg": "眉尻が中間より濃いです！尻に向かって点を減らし間隔を広げてフェードアウト。"})
        total_deduction += 10

    # === 11. 구역별 상세 ===
    zone_labels = ["眉頭(前)", "眉頭〜眉上", "眉上(中央)", "眉上〜眉尾", "眉尾(尻)"]
    for i in range(5):
        dd = stu_dark[i] - ref_dark[i]
        if abs(dd) > 20:
            feedback.append({"type": "critical", "area": zone_labels[i], "weight": 5,
                "msg": f"{'かなり濃い' if dd > 0 else 'かなり薄い'}（差{'+' if dd > 0 else ''}{dd:.0f}）"})
            total_deduction += 5
        elif abs(dd) > 12:
            feedback.append({"type": "warning", "area": zone_labels[i], "weight": 3,
                "msg": f"{'やや濃い' if dd > 0 else 'やや薄い'}（差{'+' if dd > 0 else ''}{dd:.0f}）"})
            total_deduction += 3

    # 양호 항목
    has_dark_issue = any(f["area"] in ("全体レイヤリング", "レイヤリング不足", "圧/深さ過多", "やや濃い") for f in feedback)
    if not has_dark_issue:
        feedback.insert(0, {"type": "good", "area": "全体明暗", "weight": 0,
            "msg": "全体の明暗が正解と近いです！✨"})

    score = max(0, min(100, 100 - total_deduction))

    return {
        "score": score,
        "feedback": feedback,
        "problems": problems,
        "profile_80": profile_80,
    }


def detect_pattern(zones):
    """패턴 구분 (솔직 판정)"""
    profile_80 = normalize_to_80(zones)
    avg_dark = sum(profile_80) / 5

    if avg_dark < 40:
        return {"pattern": "判別不可", "confidence": 0,
            "reason": "レイヤリングが不足でパターンの特徴が出ていません。まずレイヤリングを十分に重ねてから再提出してください。"}

    scores = {}
    for name, ref in PATTERN_REFS.items():
        diff = sum(abs(zones[i]["darkness"] - ref["darkness"][i]) for i in range(5))
        dens_diff = sum(abs(zones[i]["density"] - ref["density"][i]) for i in range(5))
        scores[name] = diff + dens_diff * 0.5

    best = min(scores, key=scores.get)
    second = sorted(scores.values())[1]
    gap = second - scores[best]

    if gap < 10:
        return {"pattern": "判別不可", "confidence": 0,
            "reason": "SOFTとMIXが似た数値です。どのパターンで作業されたか教えていただけますか？どの部分が難しかったかも教えてください。"}

    confidence = min(100, int(gap * 3))
    return {"pattern": best, "confidence": confidence,
        "reason": f"{best}パターンと判断（類似度{confidence}%）"}


def analyze_image(image_path):
    """메인 분석 함수 — 파일 경로 받음"""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "画像を読み取れません。もう一度送ってください🙏"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brows = detect_brows(image)

    if not brows:
        return {"error": "眉毛を検出できませんでした。眉毛がはっきり見える写真でもう一度送ってください🙏"}

    results = []
    for idx, brow in enumerate(brows):
        zones = analyze_zones(gray, brow)
        pat = detect_pattern(zones)
        target = pat["pattern"] if pat["pattern"] in PATTERN_REFS else "SOFT"
        analysis = generate_feedback_v2(zones, target)

        results.append({
            "brow_index": idx + 1,
            "pattern": pat,
            "analysis": analysis,
        })

    return {
        "brow_count": len(brows),
        "results": results,
    }


def analyze_image_bytes(image_bytes):
    """바이트 데이터로 분석 — app.py에서 직접 호출용"""
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


import time  # analyze_image_bytes에서 사용


def format_line_message(result, lang="ja"):
    """LINE 전송용 메시지 포맷"""
    if "error" in result:
        return f"❌ {result['error']}"

    lines = []
    if lang == "ja":
        lines.append("ご提出ありがとうございます！🙇")
        lines.append("添削させていただきます。\n")
    else:
        lines.append("제출 감사합니다! 🙇\n")

    for r in result["results"]:
        a = r["analysis"]
        score = a["score"]
        pat = r["pattern"]

        if result["brow_count"] > 1:
            lines.append(f"━━ 眉{r['brow_index']} ({pat.get('pattern', '?')}) ━━")

        lines.append(f"📊 スコア: {score}/100\n")

        good = [f for f in a["feedback"] if f["type"] == "good"]
        for f in good:
            lines.append(f"✨ {f['msg']}")

        problems = [f for f in a["feedback"] if f["type"] in ("critical", "warning")]
        if problems:
            lines.append("\n【改善ポイント】")
            for f in problems:
                icon = "🔴" if f["type"] == "critical" else "🟡"
                lines.append(f"{icon} {f['area']}: {f['msg']}")

        # 정착 관점 추가
        if any(f["area"] in ("圧/深さ過多", "ムラ/不均一", "点の間隔が詰まりすぎ") for f in problems):
            lines.append("\n💡 定着の観点：現在の状態だと圧が均一でないため、定着後にまだらに残る可能性があります。均一な密度と明暗で施術すれば定着後80%以上きれいに残ります！")

        lines.append("")

    if lang == "ja":
        lines.append("引き続き練習を頑張ってください！☺️")
    else:
        lines.append("계속 연습 화이팅! ☺️")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_image(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n--- LINE Message ---")
        print(format_line_message(result, "ja"))
    else:
        print("Usage: python image_analyzer.py <image_path>")
