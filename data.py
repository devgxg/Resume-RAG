# Manually structured chunks from Dev's resume.
# Each dict = one chunk. "section" and "title" are metadata we can filter/display later.
# This is structure-aware chunking: one chunk per logical unit (job, project, etc.)
# NOT fixed-size chunking. Contact info is deliberately its own chunk (see reasoning
# from earlier: mixing unrelated topics in one chunk dilutes the embedding for both).

CHUNKS = [
    {
        "id": "exp_dee",
        "section": "Experience",
        "title": "AI Engineer at Dee Development Engineering Ltd",
        "text": """AI Engineer, Dee Development Engineering Ltd (Dee Piping), Palwal Haryana, March 2026 - Present.
Engineered the Automated Inventory Code Allocation (AICD) system, a full-stack AI application using FastAPI, React, and LLM that dynamically matches complex MTO requirements against unallocated inventory using ASME B36.10/B36.19 logic, thickness interpolation, and step-down dimension algorithms.
Building an end-to-end AI pipeline that extracts structured Bill of Materials (BOM) from complex isometric piping drawing PDFs using Claude Vision API, pdf2image, and openpyxl, processing multiple project drawings.
Developed an ICD Code Generator: automated 19-character piping component code lookup via LLM-based tool and deterministic master sheet querying, reducing hours of manual Excel work to seconds."""
    },
    {
        "id": "exp_wesee",
        "section": "Experience",
        "title": "AI/ML Intern at WESEE, Indian Navy",
        "text": """AI/ML Intern, WESEE, Indian Navy, New Delhi, June 2025 - August 2025.
Built IMInsight, an open-source offline-capable AI image analysis system using React and Flask to support naval officers in object detection, threat identification, and anomaly detection in defense surveillance imagery.
Fine-tuned YOLOv8 and CNN-based models for high-accuracy detection of ships, personnel, radars, and aircraft. Integrated CLIP and BLIP models to generate contextual image captions from surveillance feeds.
Implemented NLP pipelines to convert visual outputs into structured, human-readable intelligence summaries for military-grade use cases.
Designed and evaluated model performance across defense-specific edge cases including low-resolution satellite imagery and partially occluded targets, improving classification robustness for real-world naval scenarios."""
    },
    {
        "id": "proj_gigscore",
        "section": "Projects",
        "title": "GigScore - AI Credit Scoring Platform",
        "text": """GigScore, built with React, Vite, TailwindCSS, Flask, Python, GROQ AI, Pandas.
Built an AI-powered alternative credit scoring platform for India's gig economy, such as Zomato riders and freelancers, generating a dynamic GigScore up to 850 by analyzing platform payouts, transaction trends, and expense ratios without relying on CIBIL scores.
Integrated a Groq AI Financial Coach that delivers personalized, actionable advice in English/Hinglish based on each worker's exact financial profile.
Engineered a Python synthetic data engine generating 90-day realistic transaction profiles for multiple risk levels. Presented at MasterX Hackathon."""
    },
    {
        "id": "proj_bitewise",
        "section": "Projects",
        "title": "BiteWise - AI Food Monitoring Assistant",
        "text": """BiteWise, AI Food Monitoring Assistant, built with Python, React, Flask, Gemini LLM, SQLite.
Built a multimodal AI web app using Gemini LLM with vision to analyze food images, delivering health scores, allergen detection, and disease-risk insights via structured JSON responses.
Engineered a Flask backend with model fallback chain, retry logic, and prompt engineering for reliable inference. Designed a React frontend with real-time camera capture and animated health dashboards.
Deployed via CI/CD pipelines, frontend on Netlify and backend on Render. Maintained code quality with Pytest test suites and version control via Git."""
    },
    {
        "id": "proj_iminsight",
        "section": "Projects",
        "title": "IMInsight - Defense AI Image Analysis",
        "text": """IMInsight, Defense AI Image Analysis, built with Python, YOLOv8, BLIP, CLIP, Flask.
Developed a defense-grade AI surveillance system enabling object detection, anomaly recognition, and multi-class classification in satellite and surveillance imagery.
Fine-tuned YOLOv8 and CNN models for high-speed detection of ships, personnel, radars, and aircraft. Integrated BLIP for NLP-based image captioning, converting visual data into actionable intelligence.
Designed a modular Flask ML pipeline for API-driven image uploads and real-time inference, built to operate fully offline for secure defense environments."""
    },
    {
        "id": "proj_heart",
        "section": "Projects",
        "title": "HeartDiseaseDetective - ML Classification Pipeline",
        "text": """HeartDiseaseDetective, built with Python, Scikit-learn, Pandas, Matplotlib.
Built an end-to-end ML classification pipeline for heart disease risk prediction using clinical patient data including cholesterol, blood pressure, and ECG features.
Evaluated multiple algorithms including Logistic Regression, Random Forest, and SVM with cross-validation, selecting the best-performing model based on precision, recall, and F1 metrics.
Performed feature engineering and correlation analysis to identify key clinical predictors, improving model interpretability for healthcare applications."""
    },
    {
        "id": "skills",
        "section": "Skills",
        "title": "Technical Skills",
        "text": """AI/LLM Engineering: LLMs including Claude, OpenAI, Gemini, Ollama, Prompt Engineering, RAG Pipelines, LangChain, AI Agents, Fine-tuning AI models.
Computer Vision: YOLOv8, OpenCV, BLIP, CLIP, CNNs, Object Detection, Image Classification.
ML/DL Frameworks: PyTorch, TensorFlow, Scikit-learn, Keras, Pandas, NumPy.
Backend and APIs: Python, FastAPI, REST APIs, MySQL, MongoDB, SQLite.
Frontend: React, JavaScript, HTML5, CSS3, TailwindCSS, Vite.
Tools and DevOps: Git, GitHub, Docker basics, Postman, VS Code, Google Colab, MS Office."""
    },
    {
        "id": "education",
        "section": "Education",
        "title": "Education",
        "text": """Amity University Haryana, Gurugram, Haryana. B.Tech in CSE, Minor in Economics, 2022 - 2026, CGPA 8.1.
Vaish Public School, Rohtak, Haryana. Class XII, CBSE, 2021, 90.8 percent.
Vaish Public School, Rohtak, Haryana. Class X, CBSE, 2019, 94.6 percent."""
    },
    {
    "id": "achievements",
    "section": "Achievements & Certifications",
    "title": "Achievements, Leadership and Certifications",
    "text": """MLSA Beta: Microsoft Learn Student Ambassador (Beta) by Microsoft.
President of MLSAxAUH Tech Society, Amity University Haryana — a leadership role leading and organizing the university's tech community.
Hackathons: Top 10 at Hackverse, IILM University Haryana. Participated among 100 teams and reached finals at MasterX Hackathon. Participated in Triwizardathon and reached top 50 qualifying teams.
Social Impact: Launched a fundraising campaign for Pawzz animal welfare organization.
Certifications: SWAYAM Python Programming, NPTEL Foundations of Virtual Reality and Augmented Reality, McKinsey Forward Graduate Program."""
    },
    {
        "id": "contact",
        "section": "Contact",
        "title": "Contact Information",
        "text": """Dev Garg. Phone: 9034729687. Email: devg55030@gmail.com. LinkedIn: linkedin.com/in/devgxg. GitHub: github.com/devgxg. Portfolio website available."""
    },
]

if __name__ == "__main__":
    print(f"Total chunks: {len(CHUNKS)}")
    for c in CHUNKS:
        print(f"  [{c['section']}] {c['title']} ({len(c['text'].split())} words)")