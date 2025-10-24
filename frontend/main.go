package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

const (
	APIBaseURL = "http://localhost:8000"
)

type PredictionRequest struct {
	Text string `json:"text"`
}

type PredictionResponse struct {
	Text          string             `json:"text"`
	Sentiment     string             `json:"sentiment"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

type BatchPredictionRequest struct {
	Reviews []string `json:"reviews"`
}

type BatchPredictionResponse struct {
	Predictions    []PredictionResponse `json:"predictions"`
	TotalCount     int                  `json:"total_count"`
	ProcessingTime float64              `json:"processing_time_ms"`
}

type HealthResponse struct {
	Status      string `json:"status"`
	ModelLoaded bool   `json:"model_loaded"`
	Device      string `json:"device"`
	Timestamp   string `json:"timestamp"`
}

type PageData struct {
	Health        *HealthResponse
	Prediction    *PredictionResponse
	Batch         *BatchPredictionResponse
	Error         string
	TestText      string
	TestReviews   string
	DatasetSample []DatasetRow
	TrainingLog   string
	TestLog       string
}

type DatasetRow struct {
	Review    string
	Sentiment string
}

var (
	trainingMux    sync.Mutex
	trainingActive bool
	trainingLog    string
	testingMux     sync.Mutex
	testingActive  bool
	testingLog     string
)

func main() {
	// Serve static files from the page directory
	fs := http.FileServer(http.Dir("../page"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))

	// Serve result images
	http.Handle("/results/", http.StripPrefix("/results/", http.FileServer(http.Dir("../results"))))

	// Routes
	http.HandleFunc("/", handleHome)
	http.HandleFunc("/predict", handlePredict)
	http.HandleFunc("/batch", handleBatch)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/dataset", handleDataset)
	http.HandleFunc("/train", handleTrain)
	http.HandleFunc("/train/log", handleTrainingLog)
	http.HandleFunc("/test", handleTest)
	http.HandleFunc("/test/log", handleTestLog)

	log.Println("Frontend server starting on http://localhost:3000")
	log.Println("Make sure FastAPI backend is running on http://localhost:8000")
	log.Fatal(http.ListenAndServe(":3000", nil))
}

func handleHome(w http.ResponseWriter, r *http.Request) {
	data := PageData{}

	// Check API health
	health, err := checkHealth()
	if err != nil {
		data.Error = "Cannot connect to API. Make sure it's running on port 8000"
	} else {
		data.Health = health
	}

	renderTemplate(w, data)
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	text := r.FormValue("text")
	data := PageData{TestText: text}

	if text == "" {
		data.Error = "Please enter a review"
		renderTemplate(w, data)
		return
	}

	// Call API
	prediction, err := predictSentiment(text)
	if err != nil {
		data.Error = "Prediction failed: " + err.Error()
	} else {
		data.Prediction = prediction
	}

	// Get health status
	data.Health, _ = checkHealth()

	renderTemplate(w, data)
}

func handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	reviews := r.FormValue("reviews")
	data := PageData{TestReviews: reviews}

	if reviews == "" {
		data.Error = "Please enter reviews (one per line)"
		renderTemplate(w, data)
		return
	}

	// Split reviews by newline
	var reviewList []string
	for _, line := range splitLines(reviews) {
		if line != "" {
			reviewList = append(reviewList, line)
		}
	}

	if len(reviewList) == 0 {
		data.Error = "No valid reviews found"
		renderTemplate(w, data)
		return
	}

	// Call API
	batch, err := predictBatch(reviewList)
	if err != nil {
		data.Error = "Batch prediction failed: " + err.Error()
	} else {
		data.Batch = batch
	}

	// Get health status
	data.Health, _ = checkHealth()

	renderTemplate(w, data)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	health, err := checkHealth()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func handleDataset(w http.ResponseWriter, r *http.Request) {
	limit := 20
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := fmt.Sscanf(limitStr, "%d", &limit); err == nil && l == 1 {
			if limit > 1000 {
				limit = 1000 // Max limit
			}
		}
	}

	sample, err := loadDatasetSample(limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	data := PageData{
		DatasetSample: sample,
	}
	data.Health, _ = checkHealth()

	renderTemplate(w, data)
}

func handleTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	trainingMux.Lock()
	if trainingActive {
		trainingMux.Unlock()
		http.Error(w, "Training already in progress", http.StatusConflict)
		return
	}
	trainingActive = true
	trainingLog = ""
	trainingMux.Unlock()

	go runTraining()

	http.Redirect(w, r, "/?training=started", http.StatusSeeOther)
}

func handleTrainingLog(w http.ResponseWriter, r *http.Request) {
	trainingMux.Lock()
	defer trainingMux.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"active": trainingActive,
		"log":    trainingLog,
	})
}

func handleTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	testingMux.Lock()
	if testingActive {
		testingMux.Unlock()
		http.Error(w, "Tests already in progress", http.StatusConflict)
		return
	}
	testingActive = true
	testingLog = ""
	testingMux.Unlock()

	go runTests()

	http.Redirect(w, r, "/?testing=started", http.StatusSeeOther)
}

func handleTestLog(w http.ResponseWriter, r *http.Request) {
	testingMux.Lock()
	defer testingMux.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"active": testingActive,
		"log":    testingLog,
	})
}

func checkHealth() (*HealthResponse, error) {
	resp, err := http.Get(APIBaseURL + "/health")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, err
	}
	return &health, nil
}

func predictSentiment(text string) (*PredictionResponse, error) {
	reqBody := PredictionRequest{Text: text}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(APIBaseURL+"/predict", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var prediction PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		return nil, err
	}
	return &prediction, nil
}

func predictBatch(reviews []string) (*BatchPredictionResponse, error) {
	reqBody := BatchPredictionRequest{Reviews: reviews}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(APIBaseURL+"/predict/batch", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var batch BatchPredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, err
	}
	return &batch, nil
}

func splitLines(s string) []string {
	var lines []string
	var line []rune
	for _, r := range s {
		if r == '\n' || r == '\r' {
			if len(line) > 0 {
				lines = append(lines, string(line))
				line = line[:0]
			}
		} else {
			line = append(line, r)
		}
	}
	if len(line) > 0 {
		lines = append(lines, string(line))
	}
	return lines
}

func loadDatasetSample(limit int) ([]DatasetRow, error) {
	file, err := os.Open("../archive/IMDB Dataset.csv")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var rows []DatasetRow

	// Skip header
	_, err = reader.Read()
	if err != nil {
		return nil, err
	}

	count := 0
	for count < limit {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if len(record) >= 2 {
			review := record[0]
			// Don't truncate - show full review
			rows = append(rows, DatasetRow{
				Review:    review,
				Sentiment: record[1],
			})
			count++
		}
	}

	return rows, nil
}

func runTraining() {
	defer func() {
		trainingMux.Lock()
		trainingActive = false
		trainingMux.Unlock()
	}()

	cmd := exec.Command("python3", "train.py")
	cmd.Dir = ".."

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		trainingMux.Lock()
		trainingLog += fmt.Sprintf("\nError creating stdout pipe: %v\n", err)
		trainingMux.Unlock()
		return
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		trainingMux.Lock()
		trainingLog += fmt.Sprintf("\nError creating stderr pipe: %v\n", err)
		trainingMux.Unlock()
		return
	}

	if err := cmd.Start(); err != nil {
		trainingMux.Lock()
		trainingLog += fmt.Sprintf("\nError starting training: %v\n", err)
		trainingMux.Unlock()
		return
	}

	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			trainingMux.Lock()
			trainingLog += line + "\n"
			trainingMux.Unlock()
		}
	}()

	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			trainingMux.Lock()
			trainingLog += line + "\n"
			trainingMux.Unlock()
		}
	}()

	cmd.Wait()

	trainingMux.Lock()
	trainingLog += "\n=== Training Complete ===\n"
	trainingMux.Unlock()
}

func runTests() {
	defer func() {
		testingMux.Lock()
		testingActive = false
		testingMux.Unlock()
	}()

	// Wait for API to be ready
	time.Sleep(2 * time.Second)

	cmd := exec.Command("python3", "test_api.py")
	cmd.Dir = ".."

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		testingMux.Lock()
		testingLog += fmt.Sprintf("\nError creating stdout pipe: %v\n", err)
		testingMux.Unlock()
		return
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		testingMux.Lock()
		testingLog += fmt.Sprintf("\nError creating stderr pipe: %v\n", err)
		testingMux.Unlock()
		return
	}

	if err := cmd.Start(); err != nil {
		testingMux.Lock()
		testingLog += fmt.Sprintf("\nError starting tests: %v\n", err)
		testingMux.Unlock()
		return
	}

	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			testingMux.Lock()
			testingLog += line + "\n"
			testingMux.Unlock()
		}
	}()

	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			testingMux.Lock()
			testingLog += line + "\n"
			testingMux.Unlock()
		}
	}()

	cmd.Wait()

	testingMux.Lock()
	testingLog += "\n=== Tests Complete ===\n"
	testingMux.Unlock()
}

func renderTemplate(w http.ResponseWriter, data PageData) {
	tmpl := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis - Macromill</title>
    <link rel="stylesheet" href="/static/Macromill Group_files/bootstrap.min.css">
    <link rel="stylesheet" href="/static/Macromill Group_files/font-awesome.min.css">
    <link rel="stylesheet" href="/static/Macromill Group_files/main.css">
    <link rel="stylesheet" href="/static/Macromill Group_files/footer.css">
    <link rel="stylesheet" href="/static/Macromill Group_files/topnav.css">
    <link rel="stylesheet" href="/static/Macromill Group_files/animate.css">
    <style>
        body { padding-top: 50px; background: #f5f5f5; color: #111; }
        .container { max-width: 1200px; color: #111; }
        .card { background: white; border-radius: 8px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); color: #111; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 8px; margin-bottom: 30px; }
        .status-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
        .status-healthy { background: #10b981; color: white; }
        .status-unhealthy { background: #ef4444; color: white; }
        .sentiment-positive { color: #10b981; font-weight: bold; font-size: 1.5em; }
        .sentiment-negative { color: #ef4444; font-weight: bold; font-size: 1.5em; }
        .confidence-bar { height: 30px; background: #e5e7eb; border-radius: 15px; overflow: hidden; margin: 10px 0; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #10b981, #059669); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: width 0.5s; }
        .prob-row { display: flex; justify-content: space-between; margin: 8px 0; padding: 8px; background: #f9fafb; border-radius: 4px; color: #111; }
        textarea { width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px; font-size: 14px; resize: vertical; color: #111; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; padding: 12px 30px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: transform 0.2s; color: white; }
        .btn-primary:hover { transform: translateY(-2px); }
        .alert { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .alert-danger { background: #fee2e2; color: #991b1b; border-left: 4px solid #ef4444; }
        .batch-item { border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; background: #f9fafb; border-radius: 4px; color: #111; }
        .viz-container { margin-top: 30px; }
        .viz-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        .tab-buttons { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab-button { padding: 12px 24px; border: none; background: #e5e7eb; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s; color: #111; }
        .tab-button.active { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        h2, h3, h4, p, label, td, th, div { color: #111; }
        select, input { color: #111; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header" style="background: url('/static/Macromill Group_files/bg_group.png') center/cover, linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="/static/Macromill Group_files/macromill.png" alt="Macromill" style="max-width: 200px; height: auto;">
            </div>
            <h1 style="text-align: center;">Sentiment Analysis Demo</h1>
            <p style="text-align: center;">Movie Review Sentiment Classification using DistilBERT</p>
            {{if .Health}}
            <div style="margin-top: 20px;">
                <span class="status-badge {{if .Health.ModelLoaded}}status-healthy{{else}}status-unhealthy{{end}}">
                    {{if .Health.ModelLoaded}}Model Ready{{else}}Model Not Loaded{{end}}
                </span>
                <span style="margin-left: 15px; opacity: 0.9;">Device: {{.Health.Device}}</span>
            </div>
            {{end}}
        </div>

        {{if .Error}}
        <div class="alert alert-danger">
            <strong>Error:</strong> {{.Error}}
        </div>
        {{end}}

        <div class="tab-buttons">
            <button class="tab-button active" onclick="showTab('dataset')">Dataset Sample</button>
            <button class="tab-button" onclick="showTab('train')">Run Training</button>
            <button class="tab-button" onclick="showTab('test')">Run Tests</button>
            <button class="tab-button" onclick="showTab('results')">Training Results</button>
        </div>

        <!-- Single Prediction Tab -->
        <div id="single-tab" class="tab-content">
            <div class="card">
                <h2>Test Single Review</h2>
                <form method="POST" action="/predict">
                    <textarea name="text" rows="5" placeholder="Enter a movie review...">{{.TestText}}</textarea>
                    <button type="submit" class="btn btn-primary" style="margin-top: 15px;">Analyze Sentiment</button>
                </form>

                {{if .Prediction}}
                <div style="margin-top: 30px; padding-top: 30px; border-top: 2px solid #e5e7eb;">
                    <h3>Prediction Result</h3>
                    <div class="sentiment-{{.Prediction.Sentiment}}">
                        {{if eq .Prediction.Sentiment "positive"}}POSITIVE{{else}}NEGATIVE{{end}}
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {{printf "%.1f" (mul .Prediction.Confidence 100)}}%">
                            {{printf "%.2f" (mul .Prediction.Confidence 100)}}% Confidence
                        </div>
                    </div>
                    <h4 style="margin-top: 20px;">Probability Distribution</h4>
                    <div class="prob-row">
                        <span><strong>Negative:</strong></span>
                        <span>{{printf "%.4f" (index .Prediction.Probabilities "negative")}}</span>
                    </div>
                    <div class="prob-row">
                        <span><strong>Positive:</strong></span>
                        <span>{{printf "%.4f" (index .Prediction.Probabilities "positive")}}</span>
                    </div>
                    <div style="margin-top: 20px; padding: 15px; background: #f9fafb; border-radius: 8px;">
                        <strong>Input Text:</strong><br>
                        <p style="margin-top: 10px; color: #666;">{{.Prediction.Text}}</p>
                    </div>
                </div>
                {{end}}
            </div>
        </div>

        <!-- Batch Prediction Tab -->
        <div id="batch-tab" class="tab-content">
            <div class="card">
                <h2>Test Multiple Reviews</h2>
                <form method="POST" action="/batch">
                    <textarea name="reviews" rows="8" placeholder="Enter multiple reviews (one per line)...">{{.TestReviews}}</textarea>
                    <button type="submit" class="btn btn-primary" style="margin-top: 15px;">Analyze Batch</button>
                </form>

                {{if .Batch}}
                <div style="margin-top: 30px; padding-top: 30px; border-top: 2px solid #e5e7eb;">
                    <h3>Batch Results</h3>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">{{.Batch.TotalCount}}</div>
                            <div class="metric-label">Reviews Analyzed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{printf "%.0f" .Batch.ProcessingTime}}</div>
                            <div class="metric-label">Processing Time (ms)</div>
                        </div>
                    </div>
                    {{range $index, $pred := .Batch.Predictions}}
                    <div class="batch-item">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span class="sentiment-{{$pred.Sentiment}}" style="font-size: 1.2em;">
                                {{if eq $pred.Sentiment "positive"}}POSITIVE{{else}}NEGATIVE{{end}}
                            </span>
                            <span style="font-weight: bold;">{{printf "%.2f" (mul $pred.Confidence 100)}}% confident</span>
                        </div>
                        <p style="color: #666; margin: 0;">{{$pred.Text}}</p>
                    </div>
                    {{end}}
                </div>
                {{end}}
            </div>
        </div>

        <!-- Dataset Sample Tab -->
        <div id="dataset-tab" class="tab-content active">
            <div class="card">
                <h2>Dataset Browser</h2>
                <p style="background: #f0f9ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0ea5e9;">
                    <strong>Source:</strong> <code>archive/IMDB Dataset.csv</code><br>
                    <strong>Total Records:</strong> 50,000 movie reviews<br>
                    <strong>Labels:</strong> positive (25,000) / negative (25,000)
                </p>

                <form method="GET" action="/dataset" style="margin-top: 20px;">
                    <label style="font-weight: bold; margin-right: 10px;">Number of reviews to load:</label>
                    <select name="limit" style="padding: 8px; border: 2px solid #e5e7eb; border-radius: 6px; margin-right: 10px;">
                        <option value="20">20</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                        <option value="200">200</option>
                        <option value="500">500</option>
                    </select>
                    <button type="submit" class="btn btn-primary">Load Reviews</button>
                </form>

                {{if .DatasetSample}}
                <div style="margin-top: 30px; padding: 15px; background: #ecfdf5; border-radius: 8px; border-left: 4px solid #10b981;">
                    <strong>Loaded {{len .DatasetSample}} reviews from archive/IMDB Dataset.csv</strong>
                </div>
                <div style="margin-top: 20px; overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: #f9fafb; border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 12px; text-align: left; width: 5%;">#</th>
                                <th style="padding: 12px; text-align: left; width: 75%;">Review Text</th>
                                <th style="padding: 12px; text-align: center;">Sentiment</th>
                            </tr>
                        </thead>
                        <tbody>
                            {{range $index, $row := .DatasetSample}}
                            <tr style="border-bottom: 1px solid #e5e7eb;">
                                <td style="padding: 12px; color: #999;">{{add $index 1}}</td>
                                <td style="padding: 12px; color: #666; font-size: 14px;">{{$row.Review}}</td>
                                <td style="padding: 12px; text-align: center;">
                                    <span style="padding: 6px 12px; border-radius: 12px; font-weight: bold; {{if eq $row.Sentiment "positive"}}background: #d1fae5; color: #065f46;{{else}}background: #fee2e2; color: #991b1b;{{end}}">
                                        {{$row.Sentiment}}
                                    </span>
                                </td>
                            </tr>
                            {{end}}
                        </tbody>
                    </table>
                </div>
                {{end}}
            </div>
        </div>

        <!-- Run Training Tab -->
        <div id="train-tab" class="tab-content">
            <div class="card">
                <h2>Run Model Training</h2>
                <p>Train the DistilBERT model on the IMDb dataset. This will take 30-60 minutes on GPU or 2-3 hours on CPU.</p>

                <form method="POST" action="/train" onsubmit="return confirm('Start training? This will take 30-60 minutes.');">
                    <button type="submit" class="btn btn-primary">Start Training</button>
                </form>

                <div style="margin-top: 30px;">
                    <h3>Training Output</h3>
                    <div id="training-log" style="background: #1e293b; color: #e2e8f0; padding: 20px; border-radius: 8px; font-family: monospace; height: 500px; overflow-y: auto; white-space: pre-wrap;">
                        Click "Start Training" to begin...
                    </div>
                </div>
            </div>
        </div>

        <!-- Run Tests Tab -->
        <div id="test-tab" class="tab-content">
            <div class="card">
                <h2>Run API Tests</h2>
                <p>Execute the test suite (test_api.py) against the running API.</p>

                <form method="POST" action="/test" onsubmit="return confirm('Start API tests?');">
                    <button type="submit" class="btn btn-primary">Run Tests</button>
                </form>

                <div style="margin-top: 30px;">
                    <h3>Test Output</h3>
                    <div id="test-log" style="background: #1e293b; color: #e2e8f0; padding: 20px; border-radius: 8px; font-family: monospace; height: 500px; overflow-y: auto; white-space: pre-wrap;">
                        Click "Run Tests" to begin...
                    </div>
                </div>
            </div>
        </div>

        <!-- Training Results Tab -->
        <div id="results-tab" class="tab-content">
            <div class="card">
                <h2>Training Results & Visualizations</h2>
                <p>These visualizations were generated during model training on the IMDb 50K dataset.</p>

                <div class="viz-container">
                    <h3>Exploratory Data Analysis</h3>
                    <img src="/results/eda_visualizations.png" alt="EDA Visualizations" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <div style="display:none; padding: 20px; background: #fef3c7; border-radius: 8px; color: #92400e;">
                        <strong>EDA visualization not found.</strong> Run <code>python3 train.py</code> to generate training results.
                    </div>
                </div>

                <div class="viz-container">
                    <h3>Model Evaluation Metrics</h3>
                    <img src="/results/evaluation_metrics.png" alt="Evaluation Metrics" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <div style="display:none; padding: 20px; background: #fef3c7; border-radius: 8px; color: #92400e;">
                        <strong>Evaluation metrics not found.</strong> Run <code>python3 train.py</code> to generate training results.
                    </div>
                </div>

                <div class="viz-container">
                    <h3>Model Performance Summary</h3>
                    <div id="metrics-container">
                        <div style="padding: 30px; background: #fef3c7; border-radius: 8px; color: #92400e; text-align: center;">
                            <strong>No training results available yet.</strong><br>
                            Run training from the "Run Training" tab to generate performance metrics.
                        </div>
                    </div>
                </div>

                <script>
                    fetch('/results/evaluation_results.json')
                        .then(r => r.json())
                        .then(data => {
                            const container = document.getElementById('metrics-container');
                            const html = '<div class="metrics">' +
                                '<div class="metric-card"><div class="metric-value">' + (data.accuracy * 100).toFixed(2) + '%</div><div class="metric-label">Accuracy</div></div>' +
                                '<div class="metric-card"><div class="metric-value">' + (data.precision * 100).toFixed(2) + '%</div><div class="metric-label">Precision</div></div>' +
                                '<div class="metric-card"><div class="metric-value">' + (data.recall * 100).toFixed(2) + '%</div><div class="metric-label">Recall</div></div>' +
                                '<div class="metric-card"><div class="metric-value">' + (data.f1 * 100).toFixed(2) + '%</div><div class="metric-label">F1 Score</div></div>' +
                                '<div class="metric-card"><div class="metric-value">' + (data.roc_auc * 100).toFixed(2) + '%</div><div class="metric-label">ROC AUC</div></div>' +
                                '</div>';
                            container.innerHTML = html;
                        })
                        .catch(() => {});
                </script>
            </div>
        </div>

        <div style="text-align: center; padding: 40px; background: white; border-radius: 8px; margin-top: 30px;">
            <img src="/static/Macromill Group_files/macromill.png" alt="Macromill" style="max-width: 150px; height: auto; margin-bottom: 20px;">
            <p style="color: #666; margin: 10px 0;">Sentiment Analysis System | DistilBERT Transformer Model</p>
            <p style="margin: 10px 0;"><a href="http://localhost:8000/docs" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">API Documentation</a></p>
            <p style="color: #999; font-size: 0.9em; margin-top: 20px;">Powered by Macromill AI Research</p>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');

            // Start polling if training or test tab
            if (tabName === 'train') {
                pollTrainingLog();
            } else if (tabName === 'test') {
                pollTestLog();
            }
        }

        let trainingPollInterval = null;
        let testPollInterval = null;

        function pollTrainingLog() {
            if (trainingPollInterval) clearInterval(trainingPollInterval);

            trainingPollInterval = setInterval(async () => {
                try {
                    const resp = await fetch('/train/log');
                    const data = await resp.json();
                    const logDiv = document.getElementById('training-log');
                    logDiv.textContent = data.log || 'No output yet...';
                    logDiv.scrollTop = logDiv.scrollHeight;

                    if (!data.active && data.log) {
                        clearInterval(trainingPollInterval);
                    }
                } catch (err) {
                    console.error('Error polling training log:', err);
                }
            }, 1000);
        }

        function pollTestLog() {
            if (testPollInterval) clearInterval(testPollInterval);

            testPollInterval = setInterval(async () => {
                try {
                    const resp = await fetch('/test/log');
                    const data = await resp.json();
                    const logDiv = document.getElementById('test-log');
                    logDiv.textContent = data.log || 'No output yet...';
                    logDiv.scrollTop = logDiv.scrollHeight;

                    if (!data.active && data.log) {
                        clearInterval(testPollInterval);
                    }
                } catch (err) {
                    console.error('Error polling test log:', err);
                }
            }, 1000);
        }

        // Check if we need to start polling on page load
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('training') === 'started') {
            showTab('train');
        } else if (urlParams.get('testing') === 'started') {
            showTab('test');
        }
    </script>
</body>
</html>`

	t := template.Must(template.New("index").Funcs(template.FuncMap{
		"mul": func(a, b float64) float64 { return a * b },
		"add": func(a, b int) int { return a + b },
	}).Parse(tmpl))

	if err := t.Execute(w, data); err != nil {
		log.Printf("Template error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
