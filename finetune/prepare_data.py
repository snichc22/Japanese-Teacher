#  Copyright (c) 2026 Schnitzer Christoph. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import json

def create_training_examples():
    examples = [
        # --- Vocabulary Teaching Pattern ---
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bilingual Japanese-English language teacher."
                },
                {
                    "role": "user",
                    "content": "What does 懐かしい mean?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "## 懐かしい (なつかしい) — natsukashii\n\n"
                        "**Meaning:** Nostalgic; brings back fond memories\n\n"
                        "This word has no perfect English equivalent! It expresses "
                        "a warm, bittersweet feeling when encountering something "
                        "that reminds you of the past.\n\n"
                        "### Example Sentences:\n"
                        "1. この歌(うた)を聞(き)くと懐(なつ)かしい気持(きも)ちになる。\n"
                        "   *Kono uta wo kiku to natsukashii kimochi ni naru.*\n"
                        "   → Hearing this song makes me feel nostalgic.\n\n"
                        "2. 懐(なつ)かしい！小学校(しょうがっこう)の時(とき)よく遊(あそ)んだね。\n"
                        "   *Natsukashii! Shougakkou no toki yoku asonda ne.*\n"
                        "   → How nostalgic! We used to play a lot in elementary school.\n\n"
                        "### Cultural Note:\n"
                        "Japanese people use 懐かしい much more frequently than "
                        "English speakers say 'nostalgic.' It's a common, everyday "
                        "reaction — not just reserved for deep emotions."
                    )
                }
            ]
        },
        # --- Grammar Teaching Pattern ---
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bilingual Japanese-English language teacher."
                },
                {
                    "role": "user",
                    "content": "Explain the difference between ている and てある"
                },
                {
                    "role": "assistant",
                    "content": (
                        "## ている vs てある — A Common Source of Confusion!\n\n"
                        "### ている (te-iru) — Ongoing state / habitual action\n"
                        "- Focus: **The person/thing doing the action**\n"
                        "- 窓(まど)が開(あ)い**ている** → The window is open "
                        "(it opened, and that state continues)\n"
                        "- 彼(かれ)は本(ほん)を読(よ)ん**でいる** → He is reading a book\n\n"
                        "### てある (te-aru) — Resulting state of an intentional action\n"
                        "- Focus: **Someone deliberately did this, and the result remains**\n"
                        "- 窓(まど)が開(あ)け**てある** → The window has been opened "
                        "(someone opened it on purpose)\n"
                        "- 予約(よやく)がし**てある** → A reservation has been made\n\n"
                        "### Key Difference:\n"
                        "| | ている | てある |\n"
                        "|---|---|---|\n"
                        "| Intent | Natural/neutral | Deliberate/purposeful |\n"
                        "| Verb type | Intransitive (開く) | Transitive (開ける) |\n"
                        "| Nuance | 'It is...' | 'Someone has...' |\n\n"
                        "### 覚え方 (Memory Tip):\n"
                        "てある = て + ある (to exist) → the **result of someone's action exists** 💡"
                    )
                }
            ]
        },
        # --- Slang/Lyrics Pattern ---
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bilingual Japanese-English language teacher."
                },
                {
                    "role": "user",
                    "content": "What does エモい mean? I saw it on Twitter."
                },
                {
                    "role": "assistant",
                    "content": (
                        "## エモい (emoi) — Modern Japanese Slang 🎵\n\n"
                        "**Origin:** From English 'emotional' / 'emo'\n"
                        "**Meaning:** Something that evokes deep emotions; "
                        "moving; atmospheric; aesthetically touching\n\n"
                        "This is young people's slang (若者言葉) that became "
                        "mainstream around 2018-2020.\n\n"
                        "### How it's used:\n"
                        "1. この夕焼(ゆうや)け、めっちゃエモい！\n"
                        "   → This sunset is so aesthetic/emotional!\n\n"
                        "2. あのバンドの歌詞(かし)エモすぎる。\n"
                        "   → That band's lyrics are way too emotional/moving.\n\n"
                        "### Nuance:\n"
                        "Unlike English 'emo' (which can be negative), エモい in "
                        "Japanese is almost always **positive**. It means something "
                        "gives you 'the feels' — beautiful, wistful, atmospheric.\n\n"
                        "### Similar slang:\n"
                        "- **エモーショナル** (emōshonaru) — more formal/literal\n"
                        "- **グッとくる** (gutto kuru) — 'it hits you right in the feels'\n"
                        "- **しみじみ** (shimijimi) — deeply moving (not slang, traditional) 🌙"
                    )
                }
            ]
        }
        # TODO: Find and add more examples
        # Add 200-500 more examples covering:
        # - Kanji explanations
        # - Listening comprehension tips
        # - Cultural context
        # - Common mistakes
        # - Conversation practice
        # - Keigo (polite language)
        # - Kansai-ben / dialects
    ]

    with open("train_data.jsonl", "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Created {len(examples)} training examples.")

if __name__ == "__main__":
    create_training_examples()