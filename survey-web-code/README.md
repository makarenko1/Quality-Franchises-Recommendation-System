# 🎬 סקר סרטים — אתר לורסל

אתר סקר רספונסיבי (מותאם לנייד) שבו המשתמש בוחר 3 סרטים מתוך `dataset.csv` ומדרג
אותם 1/2/3. התשובות נשמרות ל-Google Sheets דרך Google Apps Script.

## מבנה

```
survey-web/
├── public/
│   ├── index.html     # ה-UI (RTL, מותאם לנייד)
│   └── movies.json    # רשימת הסרטים הדחוסה (id, שם, שנה, ז'אנר)
├── api/
│   └── submit.js      # פונקציית serverless ששולחת ל-Apps Script
├── vercel.json
├── apps_script.gs     # הקוד להדבקה ב-Google Sheets
└── survey.py          # גרסת Streamlit מקומית (לא רצה על ורסל)
```

## הקמה — 3 שלבים

### 1. Google Sheets + Apps Script
1. פותחים גיליון חדש ב-[sheets.new](https://sheets.new)
2. `Extensions → Apps Script`, מדביקים את `apps_script.gs`, שומרים
3. `Deploy → New deployment → Web app`, עם `Execute as: Me` ו-`Who has access: Anyone`
4. מעתיקים את ה-Web app URL

### 2. פריסה לורסל
מתוך התיקייה הזו:
```bash
cd survey-web
vercel        # פריסת preview
vercel --prod # פריסה לפרודקשן
```

### 3. חיבור ה-URL
ב-Vercel: `Project → Settings → Environment Variables`, מוסיפים:
```
APPS_SCRIPT_URL = <ה-URL מהשלב הראשון>
```
ואז פורסים מחדש (`vercel --prod`).

## הרצה מקומית
```bash
cd survey-web
vercel dev    # מריץ גם את ה-API
```

## עדכון רשימת הסרטים
אם `dataset.csv` משתנה, מייצרים מחדש את `movies.json` מתיקיית השורש של הפרויקט:
```python
import pandas as pd, json
df = pd.read_csv("dataset.csv", usecols=["MovieID","Title","Year","Genre1"]).dropna(subset=["Title"])
yr = lambda y: int(y) if pd.notna(y) else None
data = [[int(r.MovieID), str(r.Title), yr(r.Year), (None if pd.isna(r.Genre1) else str(r.Genre1))] for r in df.itertuples()]
json.dump(data, open("survey-web/public/movies.json","w"), ensure_ascii=False, separators=(",",":"))
```
