"""
파우더브로우 이미지 분석기 v3
- 선생님 피드백 규칙 8단계 구조 적용
- 숫자와 해석 분리 / 문제 1개당 문단 1개 / 추상어 금지
- 눈썹 1개만 감지 / 초록 커팅매트 제외
"""

import cv2
import numpy as np
import os
import time
import tempfile

TEACHER = {"profile_80": [1, 28, 40, 35, 14], "gradient_range": 40}

REFS = {
    "SOFT":    {"dark": [121, 128, 136, 129, 116], "dens": [26.1, 53.0, 67.4, 67.5, 40.0]},
    "NATURAL": {"dark": [120, 122, 123, 121, 122], "dens": [25.1, 54.0, 57.4, 50.6, 35.2]},
    "MIX":     {"dark": [125, 127, 129, 128, 126], "dens": [32.1, 63.7, 62.8, 53.3, 39.9]},
}

ZONE_NAMES = ["眉頭(前)", "眉頭〜眉上", "眉上(中央)", "眉上〜眉尾", "眉尾(尻)"]


# ══════════════════════════════════════
# 감지
# ══════════════════════════════════════

def detect_brow(image):
    """눈썹 1개만 반환. 감지 우선, 필터 최소화."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 여러 threshold 시도 (밝은 사진~어두운 사진 대응)
    bg = np.median(gray)
    candidates = []

    for ratio in [0.80, 0.75, 0.70, 0.65]:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blurred, bg * ratio, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 6))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2)))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.005:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            # 눈썹 형태: 가로가 세로보다 2~8배
            if aspect < 2.0 or aspect > 8.0:
                continue
            # 높이 최소
            if bh < h * 0.03:
                continue
            candidates.append({"contour": cnt, "bbox": (x, y, bw, bh), "area": area, "ratio": ratio})

        # 후보가 있으면 더 이상 시도 안 함
        if candidates:
            break

    if not candidates:
        return None

    # 가장 큰 것 반환
    candidates.sort(key=lambda b: b["area"], reverse=True)
    best = candidates[0]
    return {"contour": best["contour"], "bbox": best["bbox"]}


def analyze_zones(gray, brow):
    x, y, w, h = brow["bbox"]
    zw = w // 5
    bg = float(np.median(gray))
    zones = []
    for i in range(5):
        zx = x + i * zw
        zwi = zw if i < 4 else (w - i * zw)
        roi = gray[y:y+h, zx:zx+zwi]
        mask = np.zeros_like(roi)
        s = brow["contour"].copy(); s[:, 0, 0] -= zx; s[:, 0, 1] -= y
        cv2.drawContours(mask, [s], -1, 255, -1)
        px = roi[mask > 0]
        if len(px) == 0:
            zones.append({"darkness": 0, "density": 0})
            continue
        dark = max(0, float(bg - np.mean(px)))
        dt = bg * 0.85
        dens = (np.sum(px < dt) / len(px)) * 100
        zones.append({"darkness": round(dark, 1), "density": round(dens, 1)})
    return zones


def to_80(zones):
    mx = max(z["darkness"] for z in zones) if zones else 1
    if mx == 0: return [0]*5
    s = 80 / mx
    return [round(z["darkness"] * s, 1) for z in zones]


# ══════════════════════════════════════
# 분석 + 피드백 생성
# ══════════════════════════════════════

def analyze_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "画像を読み取れません。もう一度送ってください🙏"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brow = detect_brow(image)
    if not brow:
        return {"error": "眉毛を検出できませんでした。眉毛がはっきり見える写真でもう一度送ってください🙏"}

    zones = analyze_zones(gray, brow)
    profile = to_80(zones)
    t = TEACHER["profile_80"]

    # 패턴 판정
    avg80 = sum(profile) / 5
    pattern = "SOFT"
    if avg80 >= 40:
        scores = {}
        for name, ref in REFS.items():
            d = sum(abs(zones[i]["darkness"] - ref["dark"][i]) for i in range(5))
            scores[name] = d
        best = min(scores, key=scores.get)
        vals = sorted(scores.values())
        pattern = best if vals[1] - vals[0] >= 10 else "SOFT"

    ref = REFS[pattern]
    stu_dens = [z["density"] for z in zones]
    ref_dens = ref["dens"]
    grad_range = max(profile) - min(profile)
    peak_idx = profile.index(max(profile))

    # 구역별 판정
    zone_results = []
    for i in range(5):
        diff = profile[i] - t[i]
        ad = abs(diff)
        icon = "✅" if ad <= 5 else "🟡" if ad <= 10 else "🔴"
        zone_results.append(f"{icon} {ZONE_NAMES[i]}：{profile[i]:.0f}（先生{t[i]}、差{diff:+.0f}）")

    # === 개선 포인트 생성 (규칙: 문제 1개당 문단 1개) ===
    points = []
    total_ded = 0

    # 1. 점 간격 과밀
    avg_dens = sum(stu_dens) / 5
    avg_ref_dens = sum(ref_dens) / 5
    if avg_dens - avg_ref_dens > 20:
        points.append(
            "🔴 点の間隔が全体的に詰まりすぎており、グラデーションが出ていません。\n"
            "特に上段は、点の間隔を意識的に広げていただき、皮膚が見える状態を作ることが重要です。\n\n"
            "👉「弱く打つ」のではなく、\n"
            "👉点の間隔を広げてレイヤリングする意識で施術してください。"
        )
        total_ded += 15
    elif avg_dens - avg_ref_dens > 10:
        points.append(
            "🟡 全体的に密度がやや高めです。\n"
            "点の間隔をもう少し広げて、皮膚が見えるようにしてみてください。"
        )
        total_ded += 5

    # 2. 앞머리 그라데이션
    stu_grad = zones[2]["darkness"] - zones[0]["darkness"]
    ref_grad = ref["dark"][2] - ref["dark"][0]
    if stu_grad < 3:
        points.append(
            "🔴 眉頭〜中央にかけてグラデーションがほとんどありません。\n"
            "眉頭から徐々に濃くなる流れを作る必要があります。\n\n"
            "・眉頭・デザイン上ライン\n"
            "　→ 点の間隔を約2mm程度まで広げて配置し、薄く見せる\n"
            "・中央（オレンジゾーン）\n"
            "　→ 回数を重ねて濃さをしっかり出す\n\n"
            "👉この明暗差を作ることで、自然な立体感が生まれます。"
        )
        total_ded += 15
    elif stu_grad < ref_grad * 0.5:
        points.append(
            "🟡 眉頭〜中央のグラデーションがもう少し必要です。\n"
            "眉頭をもう少し薄く、中央をもう少し濃くして明暗の差を広げてみてください。"
        )
        total_ded += 5

    # 3. 오렌지존 분리
    rj2 = ref["dark"][2] - ref["dark"][1]
    rj4 = ref["dark"][2] - ref["dark"][3]
    sj2 = zones[2]["darkness"] - zones[1]["darkness"]
    sj4 = zones[2]["darkness"] - zones[3]["darkness"]
    if sj2 > rj2 * 2 + 5 or sj4 > rj4 * 2 + 5:
        points.append(
            "🔴 オレンジゾーンの陰影と周囲のグラデーションが分離して見えます。\n"
            "オレンジゾーンから上/横に向かって、力を抜きながらレイヤリングをつなげてください。\n\n"
            "・ブレンディング区間を2〜3mm設けること\n"
            "・一度に仕上げようとせず、何回も薄く重ねることがポイントです。"
        )
        total_ded += 10

    # 4. 미두 뭉침
    if profile[0] > 8:
        ratio = round(profile[0] / max(t[0], 1))
        points.append(
            f"🟡 眉頭がやや濃く出ています（先生比 約{ratio}倍）。\n"
            "眉頭は「空ける」のではなく、\n"
            "間隔を広く取りながら薄く重ねることで自然に見せるのがポイントです。"
        )
        total_ded += 10

    # 4b. 미두 밀도 과다
    if len(stu_dens) > 0 and stu_dens[0] > ref_dens[0] * 1.4:
        total_ded += 5

    # 5. 전체 과다
    diff_avg = sum(zones[i]["darkness"] - ref["dark"][i] for i in range(5)) / 5
    if diff_avg > 15:
        points.append(
            "🔴 全体的に濃すぎます。\n"
            "「強く押す」のではなく、同じ弱い力で回数を重ねて濃さを出してください。\n"
            "バウンド高さ3〜5mmを意識し、赤ちゃんのほっぺを触るくらいやさしく。"
        )
        total_ded += 8

    # 6. 레이어링 부족
    if avg80 < 35:
        points.append(
            "🔴 全体的にレイヤリング回数が不足しています。\n"
            "先生は5回以上重ねていますが、この仕上がりは1〜2回程度です。\n"
            "同じ弱い力でもう3回以上重ねてください。重ねるほど「面」になります。"
        )
        total_ded += 10
    elif avg80 < 55:
        points.append(
            "🟡 レイヤリング回数がもう少し必要です。\n"
            "先生は5回以上重ねています。同じ弱い力であと2〜3回追加してください。"
        )
        total_ded += 5

    # 7. 프로파일 평탄 (전체 요약 역할도 겸함)
    if grad_range < 15:
        points.append(
            "🔴 全体の濃さが均一で、平坦な印象になっています。\n"
            "先生のデザインは\n"
            "👉 眉頭（薄い）→ 中央（最も濃い）→ 眉尾（やや薄く抜ける）\n"
            "というカーブになっていますが、\n"
            "今回の作品はその変化が弱くなっています。\n"
            "👉濃淡の差（コントラスト）を意識して調整してみてください。"
        )
        total_ded += 15
    elif grad_range < 25:
        points.append(
            "🟡 グラデーションの幅がもう少し欲しいです。\n"
            "オレンジゾーンをもう少し濃く重ねて、眉頭との差を広げてみてください。"
        )
        total_ded += 5

    # 8. 꼬리 역전
    if zones[4]["darkness"] > zones[2]["darkness"]:
        points.append(
            "🟡 眉尻が中央より濃くなっています。\n"
            "眉尻は自然にフェードアウトするように、点を減らし間隔を広げてください。"
        )
        total_ded += 5

    score = max(0, min(100, 100 - total_ded))

    return {
        "score": score,
        "avg_dark_80": round(avg80, 0),
        "grad_range": round(grad_range, 0),
        "peak_zone": ZONE_NAMES[peak_idx],
        "profile": [int(p) for p in profile],
        "zone_results": zone_results,
        "points": points,
    }


def analyze_image_bytes(image_bytes):
    tmp = os.path.join(tempfile.gettempdir(), f"brow_{int(time.time()*1000)}.jpg")
    try:
        with open(tmp, "wb") as f:
            f.write(image_bytes)
        return analyze_image(tmp)
    finally:
        try: os.remove(tmp)
        except: pass


# ══════════════════════════════════════
# LINE 메시지 포맷 (선생님 규칙 8단계)
# ══════════════════════════════════════

def format_line_message(result, context=None):
    """
    LINE 전송용 메시지. context가 있으면 맞춤 피드백 추가.
    context = {name, pattern, technique, layering, needle, practice, difficulty, improvement}
    """
    if "error" in result:
        return result["error"]

    ctx = context or {}
    t = TEACHER["profile_80"]
    p = result["profile"]
    score = result["score"]

    lines = []

    # 1. 인사 (이름 있으면 포함)
    if ctx.get("name"):
        lines.append(f"{ctx['name']}さん、ご提出ありがとうございます。")
    else:
        lines.append("ご提出ありがとうございます。")
    lines.append("🙇 添削させていただきます。")
    lines.append("")

    # 2. 정량 결과 (해석 없이 숫자만)
    lines.append("📊 分析結果")
    lines.append(f"総合スコア：{score} / 100")
    lines.append(f"濃さ：先生 80 → 学生 {result['avg_dark_80']:.0f}")
    lines.append(f"グラデーション幅：先生 40 → 学生 {result['grad_range']:.0f}")
    lines.append(f"一番濃いゾーン：{result['peak_zone']}")
    lines.append("")
    lines.append("【ゾーン別】")
    for zr in result["zone_results"]:
        lines.append(zr)
    lines.append("")
    lines.append("📈 プロファイル")
    lines.append(f"学生：{' → '.join(str(v) for v in p)}")
    lines.append(f"先生：{' → '.join(str(v) for v in t)}")

    # 3~7. 개선 포인트
    if result["points"]:
        lines.append("")
        lines.append("【改善ポイント】")
        for pt in result["points"]:
            lines.append("")
            lines.append(pt)

    # 컨텍스트 기반 추가 피드백
    extras = []

    if ctx.get("layering"):
        n = ctx["layering"]
        if n < 5:
            extras.append(f"💡 レイヤリング{n}回とのことですが、先生は5回以上重ねています。同じ弱い力であと{5-n}回以上追加してみてください。")

    if ctx.get("difficulty"):
        extras.append(f"💬 「{ctx['difficulty']}」とのこと、次回の添削で重点的に確認しますね。")

    if ctx.get("improvement"):
        extras.append(f"✨ 「{ctx['improvement']}」素晴らしいです！引き続きその調子で。")

    if ctx.get("practice"):
        n = ctx["practice"]
        if n <= 2:
            extras.append(f"📝 {n}回目の練習ですね。最初は難しくて当然です。回数を重ねるほど感覚がつかめてきます。")

    if extras:
        lines.append("")
        for e in extras:
            lines.append(e)

    # 8. 마무리
    lines.append("")
    lines.append("引き続き練習を頑張ってください。")
    lines.append("ご不明な点がございましたら、いつでもお気軽にご相談ください。☺️")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(format_line_message(analyze_image(sys.argv[1])))
    else:
        print("Usage: python image_analyzer.py <image_path>")
