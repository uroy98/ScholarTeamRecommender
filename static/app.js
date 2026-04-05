const queryInput = document.getElementById("queryInput");
const teamSize = document.getElementById("teamSize");
const recommendBtn = document.getElementById("recommendBtn");
const statusBox = document.getElementById("statusBox");
const teamSection = document.getElementById("teamSection");
const resultsContainer = document.getElementById("resultsContainer");
const rankingMode = document.getElementById("rankingMode");

function showStatus(message, kind = "info") {
  statusBox.classList.remove("hidden");
  statusBox.textContent = message;
  if (kind === "error") {
    statusBox.style.background = "#fdecec";
    statusBox.style.borderColor = "#f3b4b4";
    statusBox.style.color = "#8a1f1f";
  } else {
    statusBox.style.background = "#fff8e8";
    statusBox.style.borderColor = "#f3d298";
    statusBox.style.color = "#7a5200";
  }
}

function hideStatus() {
  statusBox.classList.add("hidden");
  statusBox.textContent = "";
}

function renderList(title, items) {
  if (!items || items.length === 0) return "";
  return `
    <div class="list-block">
      <div class="list-title">${title}</div>
      <ul>
        ${items.map(item => `<li>${item}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderTags(items) {
  if (!items || items.length === 0) return "";
  return `
    <div class="tags">
      ${items.map(item => `<span class="tag">${item}</span>`).join("")}
    </div>
  `;
}

function renderCard(scholar) {
  return `
    <article class="result-card">
      <div class="result-top">
        <div>
          <div class="result-rank">Recommended Scholar ${scholar.rank}</div>
          <h3 class="scholar-name">${scholar.name}</h3>
          <div class="affiliation">${scholar.primary_affiliation || "Affiliation not available"}</div>
          <div class="stats-row">
            <span><strong>Awards:</strong> ${scholar.award_count}</span>
            <span><strong>Papers:</strong> ${scholar.paper_count}</span>
            <span><strong>Citations:</strong> ${scholar.citation_count}</span>
            <span><strong>H-index:</strong> ${scholar.h_index}</span>
          </div>
          ${renderTags(scholar.research_keywords)}
          ${renderList("Why this scholar matches", scholar.match_reasons)}
          ${renderList("Top NSF awards", scholar.top_awards)}
          ${renderList("Top publications", scholar.top_papers)}
        </div>
        <div class="score-box">
          <div class="score-value">${scholar.score.toFixed(3)}</div>
          <div class="score-label">Relevance score</div>
        </div>
      </div>
    </article>
  `;
}

async function recommendTeam() {
  const query = queryInput.value.trim();
  if (!query) {
    showStatus("Please type a project description first.", "error");
    return;
  }

  recommendBtn.disabled = true;
  recommendBtn.textContent = "Running...";
  hideStatus();
  teamSection.classList.add("hidden");
  resultsContainer.innerHTML = "";

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        team_size: Number(teamSize.value)
      })
    });

    const data = await response.json();
    if (!response.ok) {
      showStatus(data.error || "Something went wrong.", "error");
      return;
    }

    if (data.warning) {
      showStatus(data.warning, "info");
    }

    rankingMode.textContent = `Ranking mode: ${data.ranking_mode}`;
    resultsContainer.innerHTML = data.results.map(renderCard).join("");
    teamSection.classList.remove("hidden");
  } catch (err) {
    showStatus(`Request failed: ${err.message}`, "error");
  } finally {
    recommendBtn.disabled = false;
    recommendBtn.textContent = "Recommend Team";
  }
}

recommendBtn.addEventListener("click", recommendTeam);
queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    recommendTeam();
  }
});
