import pandas as pd
import json
import os
import re
from google import genai
from collections import defaultdict

# Setup Google GenAI Client (you should set your API key in environment or replace 'YOUR_API_KEY_HERE')
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL = "gemini-2.5-flash"  # Using Gemini model

def call_llm(system_prompt, user_prompt, client):
    """Make LLM call and return response"""
    if API_KEY == "YOUR_API_KEY_HERE":
        # Return mock response if no API key is provided for testing
        return "Mock LLM Response generated because no valid API key was found."
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config={"system_instruction": system_prompt, "temperature": 0.0}
        )
        return response.text.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "LLM generation failed."

def generate_theme_narrative(theme, comments_list, client):
    sys_prompt = "Write an executive summary (3-5 sentences) for the theme based on the comments provided."
    user_prompt = f"Theme: {theme}\nComments:\n" + "\n".join([f"- {c['Sentiment']}: {c['comment']}" for c in comments_list[:15]])
    return call_llm(sys_prompt, user_prompt, client)

def generate_mom_commentary(theme, prior_pct, curr_pct, comments_list, client):
    sys_prompt = "Write a short 2-3 sentence commentary explaining the month over month shift in sentiment."
    user_prompt = f"Theme: {theme}\nPrior month negative sentiment: {prior_pct:.1f}%\nCurrent month negative sentiment: {curr_pct:.1f}%\nComments: " + "\n".join([c['comment'] for c in comments_list[:10]])
    return call_llm(sys_prompt, user_prompt, client)

def generate_mom_signals(theme, comments_list, client):
    sys_prompt = "Identify 1 key actionable signal (1 sentence) from these comments."
    user_prompt = f"Theme: {theme}\nComments: " + "\n".join([c['comment'] for c in comments_list[:10]])
    return call_llm(sys_prompt, user_prompt, client)

def generate_dashboard(df, template_path, output_path):
    if API_KEY != "YOUR_API_KEY_HERE":
        client = genai.Client(api_key=API_KEY)
    else:
        client = None
    
    # Identify Prior and Current Month (assuming sorted or we just pick top 2)
    months = sorted(df['Month'].unique())
    if len(months) >= 2:
        prior_month, current_month = months[-2], months[-1]
    elif len(months) == 1:
        prior_month, current_month = months[0], months[0]
    else:
        prior_month, current_month = "N/A", "N/A"

    df_current = df[df['Month'] == current_month]
    df_prior = df[df['Month'] == prior_month]

    dashboard_data = {
        "metadata": {
            "current_month": current_month,
            "prior_month": prior_month,
            "total_comments": len(df_current)
        },
        "theme_narratives": {},
        "comments": [],
        "mom_comparisons": {},
        "tag_explorer": []
    }

    # Populate theme narratives
    for theme in df_current['Themes'].dropna().unique():
        theme_df = df_current[df_current['Themes'] == theme]
        total = len(theme_df)
        pos = len(theme_df[theme_df['Sentiment'] == 'POSITIVE'])
        neg = len(theme_df[theme_df['Sentiment'] == 'NEGATIVE'])
        neu = len(theme_df[theme_df['Sentiment'] == 'NEUTRAL'])
        
        subtopics = theme_df['subtopics'].dropna().unique().tolist()
        
        comments_list = theme_df.to_dict('records')
        
        narrative = generate_theme_narrative(theme, comments_list, client)
        
        # We store subtopics directly instead of negative/positive_sub_themes
        dashboard_data["theme_narratives"][theme] = {
            "theme": theme,
            "total_comments": total,
            "sentiment_split": { "positive": pos, "negative": neg, "neutral": neu },
            "subtopics": [{"name": s, "approx_count": len(theme_df[theme_df['subtopics'] == s])} for s in subtopics],
            "raw_comments": {
                "positive": theme_df[theme_df['Sentiment'] == 'POSITIVE']['comment'].tolist()[:5],
                "negative": theme_df[theme_df['Sentiment'] == 'NEGATIVE']['comment'].tolist()[:5],
                "neutral": theme_df[theme_df['Sentiment'] == 'NEUTRAL']['comment'].tolist()[:5],
            },
            "narrative": narrative
        }

    # Populate MoM Comparisons
    for theme in df['Themes'].dropna().unique():
        curr_theme_df = df_current[df_current['Themes'] == theme]
        prior_theme_df = df_prior[df_prior['Themes'] == theme]
        
        curr_total = len(curr_theme_df)
        prior_total = len(prior_theme_df)
        
        if curr_total == 0 and prior_total == 0:
            continue
            
        curr_pos = len(curr_theme_df[curr_theme_df['Sentiment'] == 'POSITIVE'])
        curr_neg = len(curr_theme_df[curr_theme_df['Sentiment'] == 'NEGATIVE'])
        curr_neu = len(curr_theme_df[curr_theme_df['Sentiment'] == 'NEUTRAL'])
        
        prior_pos = len(prior_theme_df[prior_theme_df['Sentiment'] == 'POSITIVE'])
        prior_neg = len(prior_theme_df[prior_theme_df['Sentiment'] == 'NEGATIVE'])
        prior_neu = len(prior_theme_df[prior_theme_df['Sentiment'] == 'NEUTRAL'])
        
        curr_neg_pct = (curr_neg/curr_total*100) if curr_total else 0
        prior_neg_pct = (prior_neg/prior_total*100) if prior_total else 0
        
        if curr_neg_pct > prior_neg_pct + 5:
            direction = "WORSENED"
        elif curr_neg_pct < prior_neg_pct - 5:
            direction = "IMPROVED"
        else:
            direction = "STABLE"

        comments_list = curr_theme_df.to_dict('records')
        commentary = generate_mom_commentary(theme, prior_neg_pct, curr_neg_pct, comments_list, client)
        signals = generate_mom_signals(theme, comments_list, client)

        dashboard_data["mom_comparisons"][theme] = {
            "theme": theme,
            "sentiment_shift": {
                "prior": { "positive": prior_pos, "negative": prior_neg, "neutral": prior_neu },
                "current": { "positive": curr_pos, "negative": curr_neg, "neutral": curr_neu },
                "direction": direction,
                "commentary": commentary
            },
            "key_signals": signals
        }

    # Populate raw comments for drill-down
    for i, row in df.iterrows():
        dashboard_data["comments"].append({
            "response_id": f"id_{i}",
            "month": row['Month'],
            "text": row['comment'],
            "themes": [row['Themes']] if pd.notna(row['Themes']) else [],
            "tags": [row['subtopics']] if pd.notna(row['subtopics']) else [],
            "sentiment": row['Sentiment'],
            "essence": "",
            "sentiment_evidence": ""
        })

    # Generate the HTML
    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # 1. Replace Dashboard Data
    # regex to replace the const dashboardData = { ... }; block
    html = re.sub(
        r'const dashboardData = \{[\s\S]*?\};\n', 
        f'const dashboardData = {json.dumps(dashboard_data, indent=4)};\n', 
        html
    )

    # 2. Update Theme Deep Dive JS
    theme_detail_js = """
        function renderThemeDetail(theme) {
            const panel = document.getElementById('theme-detail-panel');
            const total = theme.total_comments;
            const posPct = ((theme.sentiment_split.positive / total) * 100).toFixed(1);
            const negPct = ((theme.sentiment_split.negative / total) * 100).toFixed(1);
            const neuPct = ((theme.sentiment_split.neutral / total) * 100).toFixed(1);

            let badgeClass = 'neutral';
            let badgeText = 'Healthy';
            let icon = 'fa-check-circle';
            if (negPct > 50) { badgeClass = 'negative'; badgeText = 'High Concern'; icon = 'fa-circle-exclamation'; }
            else if (negPct > 30) { badgeClass = 'warn'; badgeText = 'Moderate Concern'; icon = 'fa-triangle-exclamation'; }
            else if (posPct > 50) { badgeClass = 'positive'; badgeText = 'Excellent'; icon = 'fa-star'; }

            const subtopicsHtml = (theme.subtopics || []).map(s => `<span class="badge neutral" style="margin-right:8px; margin-bottom:8px;">${s.name} (${s.approx_count})</span>`).join('');
            
            let commentsHtml = "";
            ['negative', 'positive', 'neutral'].forEach(sent => {
                if(theme.raw_comments && theme.raw_comments[sent]) {
                    const bgClass = sent === 'negative' ? 'var(--neg-light)' : (sent === 'positive' ? 'var(--pos-light)' : 'var(--neu-light)');
                    theme.raw_comments[sent].forEach(q => {
                        commentsHtml += `<blockquote style="background:${bgClass}; padding:12px 16px; margin:12px 0; color:var(--text-main); font-style:italic; border-radius:8px; font-size:13px;">"${q}"</blockquote>`;
                    });
                }
            });

            panel.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; border-bottom:1px solid var(--border-color); padding-bottom:16px;">
                    <h2 style="font-size:1.4rem;">${theme.theme}</h2>
                    <span class="badge ${badgeClass}"><i class="fa-solid ${icon}"></i> ${badgeText}</span>
                </div>
                
                <div style="display:flex; gap:24px; margin-bottom:24px; font-family:'IBM Plex Mono', monospace; font-size:12px;">
                    <div><strong style="color:var(--text-muted);">Total Comments:</strong> ${total}</div>
                    <div style="color:var(--positive);"><strong>Positive:</strong> ${posPct}%</div>
                    <div style="color:var(--negative);"><strong>Negative:</strong> ${negPct}%</div>
                    <div style="color:var(--neutral);"><strong>Neutral:</strong> ${neuPct}%</div>
                </div>

                <div style="margin-bottom:32px; line-height:1.6; color:var(--text-main); font-size:14px; background:var(--accent-light); padding:16px; border-radius:8px; border:1px solid #C7D7FB;">
                    <strong style="color:var(--primary-accent); font-size:12px; text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:8px;">💡 AI Summary</strong>
                    ${theme.narrative}
                </div>

                <div class="grid-2">
                    <div style="background:var(--surface2); padding:20px; border-radius:8px;">
                        <h4 style="margin-bottom:16px; color:var(--text-main); font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Subtopics</h4>
                        <div style="margin-bottom:20px;">${subtopicsHtml}</div>
                    </div>
                    <div style="background:var(--surface2); padding:20px; border-radius:8px; max-height:400px; overflow-y:auto;">
                        <h4 style="margin-bottom:16px; color:var(--text-main); font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Comments</h4>
                        <div>${commentsHtml}</div>
                    </div>
                </div>
            `;
        }
"""
    # Replace the old renderThemeDetail with the new one
    html = re.sub(r'function renderThemeDetail\(theme\) \{[\s\S]*?(?=\n        // Render Comment Drill-Down)', theme_detail_js, html)

    # 3. Update Month over Month HTML structure
    mom_html_replacement = """
        <!-- TAB CONTENT: MOM -->
        <div id="mom" class="tab-content">
            <div style="display:flex; gap:16px; margin-bottom:20px; align-items:center;">
                <label style="font-size:14px; font-weight:600; color:var(--text-main);">Sentiment View:</label>
                <select id="mom-sentiment-filter" onchange="renderMomChart()" style="padding:8px 14px; border:1px solid var(--border); border-radius:8px; font-family:'Inter',sans-serif; outline:none;">
                    <option value="negative">% Negative</option>
                    <option value="positive">% Positive</option>
                    <option value="neutral">% Neutral</option>
                </select>
            </div>
            
            <h3 style="margin-bottom: 16px; font-size:0.9rem;">Key Signals & Commentary</h3>
            <div id="mom-signals-container" class="grid-2" style="margin-bottom: 24px;">
                <!-- Signal cards injected here -->
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <h3 style="margin-bottom: 20px; font-size:0.85rem; text-transform:uppercase; color:var(--text-muted); letter-spacing:1px;">
                    Prior vs Current Month Comparison</h3>
                <div style="height: 350px;">
                    <canvas id="momGroupedChart"></canvas>
                </div>
            </div>
        </div>
"""
    # Replace MoM HTML
    html = re.sub(r'<!-- TAB CONTENT: MOM -->[\s\S]*?(?=<!-- TAB CONTENT: TAGS -->)', mom_html_replacement, html)

    # 4. Remove matrix rendering from renderMomTab in JS
    html = re.sub(r'const matrixBody = document.getElementById\(\'mom-matrix-body\'\);\n', '', html)
    html = re.sub(r'const addMatrixRow[\s\S]*?\};\n\n', '', html)
    html = re.sub(r'\(mom\.new_negatives \|\| \[\]\)\.forEach\(st => addMatrixRow[\s\S]*?fa-star\'\)\);\n', '', html)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Successfully generated dashboard to {output_path}")

if __name__ == "__main__":
    # Create Mock Dataframe to test
    data = {
        'Month': ['2026-03', '2026-03', '2026-04', '2026-04', '2026-04'],
        'comment': [
            'Wait times are too long', 
            'Good service overall',
            'I waited 45 minutes on hold',
            'Jane was very helpful',
            'Portal is slow'
        ],
        'Sentiment': ['NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE'],
        'Themes': ['Customer Service', 'Customer Service', 'Customer Service', 'Customer Service', 'Digital Experience'],
        'subtopics': ['hold times', 'agent helpfulness', 'hold times', 'agent helpfulness', 'portal speed']
    }
    df = pd.DataFrame(data)
    
    template_path = r'c:\Users\harip\Documents\Python\Topic\Manual\dashboard_template_v3.html'
    output_path = r'c:\Users\harip\Documents\Python\Topic\Manual\generated_dashboard.html'
    
    generate_dashboard(df, template_path, output_path)
