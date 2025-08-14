import os, logging, re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()  # load .env variables

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CSV_PATH = "nursing_info.csv"  # Make sure this file is uploaded
TOP_K = 5
SIM_THRESHOLD = 0.45  # semantic similarity threshold

DISCLAIMER = (
    "⚕️ تنبيه مهم: البوت للمعلومات العامة فقط ومش بديل عن استشارة طبيب/ممرض مختص."
)

# -----------------------------
# Arabic normalization
# -----------------------------
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_ar(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = _ARABIC_DIACRITICS.sub("", text)
    t = re.sub("[\u0622\u0623\u0625]", "ا", t)
    t = t.replace("ى", "ي").replace("ـ", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------------
# Load KB
# -----------------------------
assert os.path.exists(CSV_PATH), f"Upload {CSV_PATH} first."
kb = pd.read_csv(CSV_PATH)
assert {"Topic", "Question", "Answer"}.issubset(kb.columns), (
    "CSV must have Topic, Question, Answer"
)
kb["q_norm"] = kb["Question"].apply(normalize_ar)

# -----------------------------
# Arabic embeddings
# -----------------------------
emb_model = SentenceTransformer("asafaya/bert-base-arabic")
kb_vec = emb_model.encode(kb["q_norm"].tolist(), normalize_embeddings=True)


# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(query: str, top_k: int = TOP_K):
    q_norm = normalize_ar(query)
    # exact match first
    exact = kb[kb["q_norm"] == q_norm]
    if not exact.empty:
        idx = exact.index[0]
        return [(idx, 1.0)]
    # semantic fallback
    q_vec = emb_model.encode([q_norm], normalize_embeddings=True)
    sims = (q_vec @ kb_vec.T).ravel()
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idxs]


# -----------------------------
# Simple answer generation without LLM
# -----------------------------
def answer_question(query: str) -> str:
    cands = retrieve(query)

    # If we have a good match, return it
    if cands and cands[0][1] >= SIM_THRESHOLD:
        idx = cands[0][0]
        row = kb.iloc[idx]
        answer = f"**{row['Topic']}**\n\n{row['Answer']}"
        return answer + "\n\n" + DISCLAIMER

    # If similarity is low, suggest similar topics
    elif cands:
        similar_topics = []
        for idx, score in cands[:3]:  # Show top 3 similar topics
            row = kb.iloc[idx]
            similar_topics.append(f"• {row['Topic']}")

        response = "لم أجد إجابة دقيقة لسؤالك. هل تقصد أحد هذه المواضيع؟\n\n"
        response += "\n".join(similar_topics)
        response += (
            "\n\nجرب إعادة صياغة السؤال أو استخدم /list لرؤية جميع المواضيع المتاحة."
        )
        return response + "\n\n" + DISCLAIMER

    # No matches at all
    else:
        return (
            "عذراً، لم أستطع العثور على إجابة لسؤالك. جرب استخدام /list لرؤية المواضيع المتاحة."
            + "\n\n"
            + DISCLAIMER
        )


# -----------------------------
# Enhanced answer generation with multiple matches
# -----------------------------
def answer_question_enhanced(query: str) -> str:
    cands = retrieve(query)

    # Collect all good matches above threshold
    good_matches = [(idx, score) for idx, score in cands if score >= SIM_THRESHOLD]

    if good_matches:
        # If we have exact match, prioritize it
        if good_matches[0][1] == 1.0:
            idx = good_matches[0][0]
            row = kb.iloc[idx]
            return f"**{row['Topic']}**\n\n{row['Answer']}\n\n" + DISCLAIMER

        # Multiple good matches - combine them
        if len(good_matches) > 1:
            response = "إليك المعلومات المتعلقة بسؤالك:\n\n"
            for i, (idx, score) in enumerate(good_matches[:2], 1):  # Limit to 2 matches
                row = kb.iloc[idx]
                response += f"**{i}. {row['Topic']}**\n{row['Answer']}\n\n"
            return response + DISCLAIMER

        # Single good match
        else:
            idx = good_matches[0][0]
            row = kb.iloc[idx]
            return f"**{row['Topic']}**\n\n{row['Answer']}\n\n" + DISCLAIMER

    # No good matches - suggest alternatives
    elif cands:
        similar_topics = []
        for idx, score in cands[:3]:
            row = kb.iloc[idx]
            similar_topics.append(f"• {row['Topic']}")

        response = "لم أجد إجابة دقيقة. هل تقصد:\n\n"
        response += "\n".join(similar_topics)
        response += "\n\n💡 جرب إعادة صياغة السؤال أو /list للمواضيع المتاحة"
        return response + "\n\n" + DISCLAIMER

    return (
        "عذراً، لم أجد معلومات متعلقة بسؤالك. استخدم /list لرؤية المواضيع المتاحة.\n\n"
        + DISCLAIMER
    )


# -----------------------------
# Self-test
# -----------------------------
def run_self_test():
    test_queries = ["ابتلاع جسم غريب", "قسطرة", "إسعافات أولية"]
    print("🧪 Running self-tests...\n")

    for query in test_queries:
        print(f"Query: {query}")
        cands = retrieve(query)
        print(f"Retrieval: {[(i, round(s, 3)) for i, s in cands[:3]]}")
        answer = answer_question_enhanced(query)
        print(f"Answer preview: {answer[:100]}...")
        print("-" * 50)


# -----------------------------
# Telegram bot handlers
# -----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """👋 أهلاً وسهلاً! أنا مساعد التمريض 🩺

أستطيع مساعدتك في:
• إجراءات التمريض
• الإسعافات الأولية
• رعاية القسطرة والكانيولا
• معلومات طبية عامة

**الأوامر المتاحة:**
/list - عرض جميع المواضيع
/help - المساعدة
/search [كلمة] - البحث في المواضيع

اكتب سؤالك وسأحاول مساعدتك! 😊"""

    await update.message.reply_text(welcome_msg, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_msg = """🆘 **المساعدة**

**كيفية الاستخدام:**
• اكتب سؤالك مباشرة
• استخدم كلمات واضحة ومحددة
• جرب مرادفات مختلفة إذا لم تجد الإجابة

**أمثلة على الأسئلة:**
• "كيف أتعامل مع ابتلاع جسم غريب؟"
• "خطوات تركيب قسطرة"
• "إسعافات أولية للجروح"

**ملاحظة مهمة:**
هذا البوت للمعلومات التعليمية فقط ولا يغني عن استشارة الطبيب المختص."""

    await update.message.reply_text(help_msg, parse_mode="Markdown")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Group topics and show them in a nice format
        topics = sorted(kb["Topic"].unique().tolist())

        # Split into chunks to avoid telegram message limits
        chunk_size = 20
        chunks = [topics[i : i + chunk_size] for i in range(0, len(topics), chunk_size)]

        for i, chunk in enumerate(chunks, 1):
            msg = f"📋 **المواضيع المتاحة ({i}/{len(chunks)}):**\n\n"
            for j, topic in enumerate(chunk, 1):
                msg += f"{i * chunk_size - chunk_size + j}. {topic}\n"

            await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text("حدث خطأ في عرض المواضيع. جرب مرة أخرى.")


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "استخدام: /search [كلمة البحث]\nمثال: /search قسطرة"
        )
        return

    search_term = " ".join(context.args)

    # Search in topics and questions
    matches = kb[
        kb["Topic"].str.contains(search_term, case=False, na=False)
        | kb["Question"].str.contains(search_term, case=False, na=False)
    ]

    if matches.empty:
        await update.message.reply_text(
            f"لم أجد نتائج للبحث عن: '{search_term}'\nجرب كلمات أخرى أو /list للمواضيع"
        )
        return

    response = f"🔍 **نتائج البحث عن '{search_term}':**\n\n"
    for _, row in matches.head(5).iterrows():  # Limit to 5 results
        response += f"• {row['Topic']}\n"

    if len(matches) > 5:
        response += f"\n... و {len(matches) - 5} نتائج أخرى"

    await update.message.reply_text(response, parse_mode="Markdown")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text or ""

    # Ignore empty messages
    if not query.strip():
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        answer = answer_question_enhanced(query)

        # Split long messages if needed
        if len(answer) > 4000:  # Telegram limit is 4096
            parts = [answer[i : i + 3500] for i in range(0, len(answer), 3500)]
            for part in parts:
                await update.message.reply_text(part, parse_mode="Markdown")
        else:
            await update.message.reply_text(answer, parse_mode="Markdown")

    except Exception as e:
        logging.exception("Error answering question")
        error_msg = (
            "🚨 حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.\n\nإذا استمر الخطأ، جرب:"
        )
        error_msg += "\n• إعادة صياغة السؤال"
        error_msg += "\n• استخدام /list للمواضيع المتاحة"
        await update.message.reply_text(error_msg)


# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# -----------------------------
# Main execution
# -----------------------------
def main():
    # Run self-test first
    try:
        run_self_test()
        print("✅ Self-test completed successfully!\n")
    except Exception as e:
        print(f"❌ Self-test failed: {e}")
        return

    # Build and configure bot
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("🤖 Bot is ready! Starting polling...")
    print("Press Ctrl+C to stop the bot")

    # Start the bot
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")


if __name__ == "__main__":
    main()
