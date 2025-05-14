const reviewElements = document.querySelectorAll('[data-hook="review"]');
let reviews = [];

reviewElements.forEach(el => {
  const text = el.innerText.trim();
  if (text) reviews.push({ text, element: el });
});

console.log(reviews)


function cleanReviewText(text) {
    let lines = text.split("\n");
    let filtered = [];

    lines.forEach(line => {
        line = line.trim();
        if (
            !line || 
            line.toLowerCase().includes("verified purchase") || 
            line.toLowerCase().includes("out of 5 stars") ||
            line.toLowerCase().startsWith("reviewed in") ||
            line.toLowerCase().startsWith("band colour") ||
            ["helpful", "report"].includes(line.toLowerCase()) ||
            /^\d+(\.\d+)? out of 5 stars/i.test(line) ||
            /^\w+\s*$/.test(line) // likely a name
        ) {
            return;
        }
        filtered.push(line);
    });

    let cleanedText = filtered.join(" ");
    cleanedText = cleanedText.replace(/[^a-zA-Z0-9\s]/g, " ");
    cleanedText = cleanedText.replace(/\s+/g, " ");
    return cleanedText.toLowerCase().trim();
}


fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ texts: reviews.map(r => r.text) })
})
.then(res => res.json())
.then(data => {
  const preds = data.predictions.map(p => p[0]); // Flatten [[0], [1]] to [0, 1]
  preds.forEach((label, i) => {
    const tag = document.createElement("div");
    tag.textContent = label ? " Fake Review" : "✅ Genuine Review";
    tag.style.color = label ? "red" : "green";
    tag.style.fontWeight = "bold";
    reviews[i].element.prepend(tag);
  });
})
.catch(err => console.error("API Error:", err));
