// Google Apps Script — מקבל תשובות מהסקר ושומר אותן בגיליון.
//
// התקנה:
// 1. פותחים Google Sheet חדש (sheets.new).
// 2. תפריט: Extensions  ->  Apps Script.
// 3. מוחקים את הקוד הקיים, מדביקים את הקוד הזה, שומרים.
// 4. Deploy -> New deployment -> Type: Web app.
//      - Execute as: Me
//      - Who has access: Anyone
// 5. מעתיקים את ה-Web app URL ומדביקים ב-APPS_SCRIPT_URL (env) של האפליקציה.
//
// בדיקה: אי אפשר לבדוק את doPost בלחיצה על Run בעורך — אין שם בקשת HTTP,
// ולכן e יהיה undefined. כדי לבדוק שהכתיבה לגיליון עובדת, הריצו את
// הפונקציה testDoPost (היא מזריקה נתוני דמה). לבדיקה מהדפדפן פתחו את
// כתובת ה-Web app — doGet יחזיר "Survey endpoint is live".

function doPost(e) {
  if (!e || !e.postData || !e.postData.contents) {
    return ContentService
      .createTextOutput(JSON.stringify({ result: "error", message: "No POST body" }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  var data = JSON.parse(e.postData.contents);
  saveRow(data);

  return ContentService
    .createTextOutput(JSON.stringify({ result: "success" }))
    .setMimeType(ContentService.MimeType.JSON);
}

// סימן גרסה — מאפשר לוודא מבחוץ (curl על ה-/exec) שהקוד החדש באמת פרוס.
function doGet() {
  return ContentService.createTextOutput("Survey endpoint is live — v2 (scores)");
}

var HEADERS = [
  "timestamp", "name",
  "rank1_movie_id", "rank1_title", "rank1_year", "rank1_score",
  "rank2_movie_id", "rank2_title", "rank2_year", "rank2_score",
  "rank3_movie_id", "rank3_title", "rank3_year", "rank3_score"
];

// כותב שורה אחת לגיליון. מוודא ששורת הכותרות קיימת ומעודכנת (גם בגיליון
// שכבר יש בו נתונים מהגרסה הישנה ללא עמודות הציון).
function saveRow(data) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADERS);
  } else {
    // אם הכותרות קצרות מהמצופה (גרסה ישנה) — מרחיבים אותן.
    var firstRow = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    if (firstRow.length < HEADERS.length || firstRow[5] !== "rank1_score") {
      sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
    }
  }

  sheet.appendRow([
    data.timestamp, data.name,
    data.rank1_movie_id, data.rank1_title, data.rank1_year, data.rank1_score,
    data.rank2_movie_id, data.rank2_title, data.rank2_year, data.rank2_score,
    data.rank3_movie_id, data.rank3_title, data.rank3_year, data.rank3_score
  ]);
}

// הריצו את זה ישירות מהעורך כדי לוודא שהכתיבה לגיליון עובדת.
function testDoPost() {
  saveRow({
    timestamp: new Date().toISOString(),
    name: "Test User",
    rank1_movie_id: 6, rank1_title: "Heat", rank1_year: 1995, rank1_score: 10,
    rank2_movie_id: 3, rank2_title: "Grumpier Old Men", rank2_year: 1995, rank2_score: 7,
    rank3_movie_id: 4, rank3_title: "Waiting to Exhale", rank3_year: 1995, rank3_score: 5
  });
}
