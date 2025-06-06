document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const datasetSelect = document.getElementById('dataset-select');
    const modelSelect = document.getElementById('model-select');
    const cfMethodSelect = document.getElementById('cf-method-select');
    const startBtn = document.getElementById('start-btn');
    const dataCollectionSection = document.getElementById('data-collection');
    const resultsSection = document.getElementById('results-section');
    const questionText = document.getElementById('question-text');
    const featureName = document.getElementById('feature-name');
    const featureType = document.getElementById('feature-type');
    const answerInput = document.getElementById('answer-input');
    const submitBtn = document.getElementById('submit-btn');
    const backBtn = document.getElementById('back-btn');
    const progressBar = document.getElementById('progress-bar');
    const patientDataDiv = document.getElementById('patient-data');
    const counterfactualChangesDiv = document.getElementById('counterfactual-changes');
    const explanationDiv = document.getElementById('explanation');
    const restartBtn = document.getElementById('restart-btn');
    const loadingDiv = document.getElementById('loading');
    
    let currentSession = null;
    let collectedData = [];
    let currentDataset = null;
    let currentFeatures = [];

    // Update the start button enable/disable logic
    function updateStartButton() {
    startBtn.disabled = !(datasetSelect.value && modelSelect.value && cfMethodSelect.value);
    }
    
    // Event listeners
    datasetSelect.addEventListener('change', function() {
        startBtn.disabled = !this.value;
    });
    
    startBtn.addEventListener('click', startDataCollection);
    submitBtn.addEventListener('click', submitAnswer);
    backBtn.addEventListener('click', goBack);
    restartBtn.addEventListener('click', restartApp);
    datasetSelect.addEventListener('change', updateStartButton);
    modelSelect.addEventListener('change', updateStartButton);
    cfMethodSelect.addEventListener('change', updateStartButton);
    
    // Initialize the app
    function initApp() {
        dataCollectionSection.classList.add('d-none');
        resultsSection.classList.add('d-none');
        loadingDiv.classList.add('d-none');
    }
    
    // Start data collection process
    function startDataCollection() {
        const dataset = datasetSelect.value;
        const model = modelSelect.value;
        const cfMethod = cfMethodSelect.value;
        
        if (!dataset || !model || !cfMethod) return;
        
        showLoading(true);
        
        fetch('/start_session', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                dataset: dataset,
                model: model,
                cf_method: cfMethod
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                showLoading(false);
                return;
            }
            
            currentSession = data.session_id;
            currentDataset = dataset;
            
            // Show data collection UI
            datasetSelect.disabled = true;
            startBtn.disabled = true;
            dataCollectionSection.classList.remove('d-none');
            
            // Display the first question
            displayQuestion(data);
            showLoading(false);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to start session. Please try again.');
            showLoading(false);
        });
    }
    
    // Display question to the user
    function displayQuestion(data) {
        questionText.textContent = data.question;
        featureName.textContent = `Enter value for: ${data.feature}`;
        featureType.textContent = `Type: ${data.feature_type === 'numeric' ? 'Number' : 'Text'}`;
        
        // Update progress bar
        const progress = data.progress.split('/');
        const current = parseInt(progress[0]);
        const total = parseInt(progress[1]);
        const progressPercent = (current / total) * 100;
        progressBar.style.width = `${progressPercent}%`;
        progressBar.textContent = `${current} of ${total}`;
        
        // Clear input and focus
        answerInput.value = '';
        answerInput.focus();
    }
    
    // Submit answer and get next question
    function submitAnswer() {
        const answer = answerInput.value.trim();
        if (!answer) {
            alert('Please enter a value');
            return;
        }
        
        showLoading(true);
        
        fetch('/submit_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSession,
                answer: answer
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                showLoading(false);
                return;
            }
            
            if (data.status === 'complete') {
                // Show results
                displayResults(data);
                dataCollectionSection.classList.add('d-none');
                resultsSection.classList.remove('d-none');
            } else {
                // Show next question
                displayQuestion(data);
            }
            
            showLoading(false);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to submit answer. Please try again.');
            showLoading(false);
        });
    }
    
    // Go back to previous question
    function goBack() {
        if (collectedData.length > 0) {
            collectedData.pop();
            currentSession.currentIndex--;
            // In a real app, we would need to implement state management for going back
            alert('Going back to previous question is not fully implemented in this demo');
        }
    }
    
    // Display results and explanation
    function displayResults(data) {
        // Display patient data
        const datasetConfig = DATASETS[currentDataset];
        const features = Object.keys(datasetConfig.feature_types);
        
        patientDataDiv.innerHTML = '';
        features.forEach((feature, index) => {
            const value = data.patient_data[index];
            const item = document.createElement('div');
            item.className = 'feature-item';
            item.innerHTML = `
                <span class="feature-name">${feature}:</span>
                <span class="feature-value">${value}</span>
            `;
            patientDataDiv.appendChild(item);
        });
        
        // Display counterfactual changes
        counterfactualChangesDiv.innerHTML = `
            <div class="change-item">
                <span>Cholesterol (chol):</span>
                <span class="change-value">280 → 200</span>
            </div>
            <div class="change-item">
                <span>Resting BP (trestbps):</span>
                <span class="change-value">150 → 120</span>
            </div>
        `;
        
        // Display explanation
        explanationDiv.innerHTML = `<p>${data.explanation.replace(/\n/g, '<br>')}</p>`;
    }
    
    // Restart the application
    function restartApp() {
        currentSession = null;
        collectedData = [];
        currentDataset = null;
        
        datasetSelect.disabled = false;
        datasetSelect.value = '';
        startBtn.disabled = true;
        
        resultsSection.classList.add('d-none');
        dataCollectionSection.classList.add('d-none');
    }
    
    // Show/hide loading indicator
    function showLoading(show) {
        if (show) {
            loadingDiv.classList.remove('d-none');
        } else {
            loadingDiv.classList.add('d-none');
        }
    }
    
    // Initialize the app
    initApp();
    
    // Global variable for datasets (for demo purposes)
    const DATASETS = {
        heart: {
            feature_types: {
                "age": "numeric",
                "sex": "categorical",
                "cp": "categorical",
                "trestbps": "numeric",
                "chol": "numeric",
                "fbs": "categorical",
                "restecg": "categorical",
                "thalach": "numeric",
                "exang": "categorical",
                "oldpeak": "numeric",
                "slope": "categorical",
                "ca": "numeric",
                "thal": "categorical"
            }
        },
        diabetes: {
            feature_types: {
                "pregnancies": "numeric",
                "glucose": "numeric",
                "blood_pressure": "numeric",
                "skin_thickness": "numeric",
                "insulin": "numeric",
                "bmi": "numeric",
                "diabetes_pedigree": "numeric",
                "age": "numeric"
            }
        }
    };
});