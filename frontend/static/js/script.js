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
    const progressText = document.getElementById('progress-text');
    const userDataDiv = document.getElementById('user-data');
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
    datasetSelect.addEventListener('change', updateStartButton);
    modelSelect.addEventListener('change', updateStartButton);
    cfMethodSelect.addEventListener('change', updateStartButton);
    
    startBtn.addEventListener('click', startDataCollection);
    submitBtn.addEventListener('click', submitAnswer);
    backBtn.addEventListener('click', goBack);
    restartBtn.addEventListener('click', restartApp);
    
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
            modelSelect.disabled = true;
            cfMethodSelect.disabled = true;
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

        document.querySelector('.progress-text span:first-child').textContent = `Getting started`;
        document.querySelector('.progress-text span:last-child').textContent = `${Math.round(progressPercent)}%`;
        progressText.textContent = `Question ${current} of ${total}`;  // Header text
        
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
        
        // Disable submit button to prevent double submission
        submitBtn.disabled = true;
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
        .then(response => {
            // Check if response is ok
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert(data.error);
                showLoading(false);
                submitBtn.disabled = false;
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
            submitBtn.disabled = false;
        })
        .catch(error => {
            console.error('Error details:', error);
            alert(`Failed to submit answer: ${error.message}. Please try again.`);
            showLoading(false);
            submitBtn.disabled = false;
        });
    }
    
    // Display results and explanation
    function displayResults(data) {
        document.getElementById('original-prediction').textContent =
            data.original_prediction_label;
        document.getElementById('counterfactual-prediction').textContent =
            data.new_prediction_label;
        
        // Display user data
        userDataDiv.innerHTML = '';
        if (data.feature_names && data.user_data) {
            data.feature_names.forEach((feature, index) => {
                const value = data.user_data[index];
                const item = document.createElement('div');
                item.className = 'feature-item';
                item.innerHTML = `
                    <span class="feature-name">${feature}:</span>
                    <span class="feature-value">${value}</span>
                `;
                userDataDiv.appendChild(item);
            });
        }
        
        // Display counterfactual changes
        counterfactualChangesDiv.innerHTML = '';
        if (data.required_changes && Array.isArray(data.required_changes)) {
            data.required_changes.forEach(change => {
                const changeItem = document.createElement('div');
                changeItem.className = 'change-item';
                changeItem.innerHTML = `
                    <span>${change.feature}:</span>
                    <span class="change-value"> ${change.new}</span>
                `;
                counterfactualChangesDiv.appendChild(changeItem);
            });
        } else {
            counterfactualChangesDiv.innerHTML = '<p>No specific changes required.</p>';
        }
        
        // Display explanation
        explanationDiv.innerHTML = `<p>${data.explanation.replace(/\n/g, '<br>')}</p>`;
    }
    
    // Go back function (placeholder)
    function goBack() {
        // This would need backend support to implement properly
        alert('Back functionality not implemented yet');
    }
    
    // Restart the application
    function restartApp() {
        currentSession = null;
        collectedData = [];
        currentDataset = null;
        
        datasetSelect.disabled = false;
        modelSelect.disabled = false;
        cfMethodSelect.disabled = false;
        datasetSelect.value = '';
        modelSelect.value = '';
        cfMethodSelect.value = '';
        updateStartButton();
        
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
});