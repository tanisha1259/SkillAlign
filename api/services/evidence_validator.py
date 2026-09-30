from __future__ import annotations
import re
def _normalise(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)
def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-zA-Z][a-zA-Z0-9+#.-]",
            text,
        )
        if len(token) >= 4
    }
def validate_evidence(
    requirement: str,
    requirement_type: str,
    evidence: str,
) -> bool:
    req = _normalise(requirement)
    ev = _normalise(evidence)
    if not ev:
        return False
    # ---------------------------------------------------------
    # EDUCATION
    # ---------------------------------------------------------
    # Education is handled by compare_education().
    if requirement_type == "education":
        return False
    # ---------------------------------------------------------
    # TOOLS
    # ---------------------------------------------------------
    if requirement_type == "tool":
        tool_groups = {
            "salesforce": ["salesforce"],
            "hubspot": ["hubspot"],
            "microsoft office": [
                "microsoft office",
                "ms office",
                "excel",
                "microsoft excel",
                "word",
                "microsoft word",
                "powerpoint",
                "microsoft powerpoint",
            ],
            "power bi": ["power bi"],
            "tableau": ["tableau"],
            "sap": ["sap"],
            "oracle": ["oracle"],
            "jira": ["jira"],
            "confluence": ["confluence"],
            "python": ["python"],
            "sql": ["sql"],
            "java": ["java"],
            "c++": ["c++"],
            "aws": [
                "aws",
                "amazon web services",
            ],
            "azure": [
                "azure",
            ],
            "gcp": [
                "gcp",
                "google cloud",
            ],
            "docker": [
                "docker",
            ],
        }
        required_tools = []
        for tool, aliases in tool_groups.items():
            if tool in req:
                required_tools.extend(aliases)
        if required_tools:
            return _contains_any(ev, required_tools)
    # ---------------------------------------------------------
    # EXPERIENCE
    # ---------------------------------------------------------
    if requirement_type == "experience":
        # -----------------------------------------------------
        # Cloud platform EXPERIENCE
        # -----------------------------------------------------
        # Specific database EXPERIENCE
        # -----------------------------------------------------
        # Generic SQL evidence must not satisfy a requirement for a
        # specific database engine such as PostgreSQL or MySQL.
        database_required = (
            "postgresql" in req
            or "postgres" in req
            or "mysql" in req
            or "microsoft sql server" in req
            or "sql server" in req
        )
        if database_required:
            database_aliases = [
                "postgresql",
                "postgres",
                "mysql",
                "microsoft sql server",
                "sql server",
            ]
            return _contains_any(ev, database_aliases)
        # -----------------------------------------------------
        # Statistical analysis EXPERIENCE
        # -----------------------------------------------------
        if "statistical analysis" in req or "statistical" in req or "statistics" in req:
            statistical_aliases = [
                "statistical analysis",
                "statistical modeling",
                "statistical modelling",
                "hypothesis testing",
                "descriptive statistics",
                "inferential statistics",
                "statistical testing",
                "regression analysis",
            ]
            return _contains_any(ev, statistical_aliases)
        # -----------------------------------------------------
        # Scalable backend/system-design EXPERIENCE
        # -----------------------------------------------------
        if (
            "scalable backend" in req
            or "scalable backend systems" in req
            or "scalable systems" in req
            or "scalable system" in req
            or "scalability" in req
        ):
            scalability_aliases = [
                "scalable backend",
                "scalable system",
                "scalable systems",
                "backend architecture",
                "system architecture",
                "distributed system",
                "distributed systems",
                "microservices",
                "high availability",
                "load balancing",
                "horizontal scaling",
                "vertical scaling",
                "scalability",
            ]
            return _contains_any(ev, scalability_aliases)
        # -----------------------------------------------------
        # Business-development / client-relationship EXPERIENCE
        # -----------------------------------------------------
        if (
            "generating leads" in req
            or "generate leads" in req
            or "lead generation" in req
            or "client relationships" in req
            or "client relationship" in req
            or "managing client relationships" in req
        ):
            business_aliases = [
                "lead generation",
                "generate leads",
                "generated leads",
                "generating leads",
                "client relationship",
                "client relationships",
                "client management",
                "account management",
                "client acquisition",
                "customer relationship",
                "customer management",
                "managed clients",
            ]
            return _contains_any(ev, business_aliases)
        # -----------------------------------------------------
        # A requirement such as:
        # "Experience with AWS"
        # must have actual AWS evidence.
        #
        # This is intentionally separate from the generic
        # technical-skill/tool handling.
        cloud_required = (
            "aws" in req
            or "amazon web services" in req
            or "azure" in req
            or "microsoft azure" in req
            or "gcp" in req
            or "google cloud" in req
        )
        if cloud_required:
            cloud_aliases = [
                "aws",
                "amazon web services",
                "azure",
                "microsoft azure",
                "gcp",
                "google cloud",
            ]
            return _contains_any(ev, cloud_aliases)
        # -----------------------------------------------------
        # Docker deployment/containerization EXPERIENCE
        # -----------------------------------------------------
        # Merely listing "Docker" in a technical-skills section
        # is not sufficient for an experience requirement such as:
        #
        # "Experience deploying ML applications using Docker."
        if "docker" in req:
            deployment_aliases = [
                "dockerized",
                "dockerised",
                "containerized",
                "containerised",
                "containerization",
                "containerisation",
                "docker container",
                "deployed using docker",
                "deployed with docker",
                "deployed docker",
                "docker deployment",
                "dockerized application",
                "dockerised application",
                "dockerized applications",
                "dockerised applications",
            ]
            return _contains_any(ev, deployment_aliases)
        # -----------------------------------------------------
        # Specific technology EXPERIENCE
        # -----------------------------------------------------
        technology_aliases = {
            "kubernetes": [
                "kubernetes",
                "k8s",
            ],
            "tensorflow": [
                "tensorflow",
            ],
            "pytorch": [
                "pytorch",
            ],
            "fastapi": [
                "fastapi",
            ],
            "flask": [
                "flask",
            ],
            "numpy": [
                "numpy",
            ],
            "pandas": [
                "pandas",
            ],
            "scikit-learn": [
                "scikit-learn",
                "sklearn",
            ],
            "sql": [
                "sql",
            ],
        }
        for technology, aliases in technology_aliases.items():
            if technology in req:
                if not _contains_any(ev, aliases):
                    return False
        # -----------------------------------------------------
        # Domain experience
        # -----------------------------------------------------
        domain_groups = {
            "business development": [
                "business development",
                "sales development",
                "lead generation",
                "client acquisition",
                "account management",
                "client management",
            ],
            "sales": [
                "sales",
                "business development",
                "account management",
                "client acquisition",
            ],
            "marketing": [
                "marketing",
                "digital marketing",
                "brand marketing",
            ],
            "software development": [
                "software development",
                "software engineer",
                "software developer",
                "developer",
                "programming",
            ],
            "machine learning": [
                "machine learning",
                "deep learning",
                "ml",
            ],
            "data science": [
                "data science",
                "data scientist",
            ],
            "project management": [
                "project management",
                "project manager",
                "program manager",
            ],
        }
        domains = []
        for domain, aliases in domain_groups.items():
            if domain in req:
                domains.extend(aliases)
        # Requirement asks for a specific domain.
        if domains and not _contains_any(ev, domains):
            return False
        # -----------------------------------------------------
        # Leadership / mentoring requirements
        # -----------------------------------------------------
        leadership_words = [
            "lead",
            "led",
            "leader",
            "leadership",
            "manage",
            "managed",
            "manager",
            "mentored",
            "mentoring",
            "supervised",
            "supervising",
        ]
        mentoring_required = any(
            word in req
            for word in [
                "leading",
                "leadership",
                "managing",
                "management",
                "mentoring",
                "mentor",
                "supervising",
            ]
        )
        if mentoring_required:
            if not _contains_any(ev, leadership_words):
                return False
        return True
    # ---------------------------------------------------------
    # CAPABILITIES
    # ---------------------------------------------------------
    if requirement_type == "capability":
        capability_groups = {
            "client relationships": [
                "client relationship",
                "client management",
                "customer relationship",
                "account management",
                "client engagement",
                "managed clients",
            ],
            "generate leads": [
                "generate leads",
                "generated leads",
                "lead generation",
                "leads",
            ],
            "multiple projects": [
                "multiple projects",
                "managed projects",
                "managed multiple",
                "coordinated multiple",
                "projects simultaneously",
                "simultaneously",
            ],
            "analytical": [
                "analysis",
                "analytical",
                "analytics",
                "data analysis",
            ],
            "strategic": [
                "strategic",
                "strategy",
                "strategic planning",
            ],
        }
        matched_aliases = []
        for phrase, aliases in capability_groups.items():
            if phrase in req:
                matched_aliases.extend(aliases)
        if matched_aliases:
            return _contains_any(ev, matched_aliases)
    # ---------------------------------------------------------
    # SOFT SKILLS
    # ---------------------------------------------------------
    if requirement_type == "soft_skill":
        soft_skill_groups = {
            "communication": [
                "communicated with",
                "communicated",
                "presented findings",
                "presented to",
                "gave presentations",
                "delivered presentations",
                "public speaking",
                "interpersonal",
                "stakeholder communication",
                "client communication",
                "written communication",
                "oral communication",
                "coordinated events",
                "coordinated with",
                "collaborated with",
            ],
            "leadership": [
                "leadership",
                "led",
                "managed",
                "mentored",
                "mentoring",
                "team lead",
                "team leader",
                "supervised",
            ],
            "teamwork": [
                "teamwork",
                "team player",
                "collaborated",
                "collaboration",
                "worked with the team",
            ],
        }
        matched_aliases = []
        for skill, aliases in soft_skill_groups.items():
            if skill in req:
                matched_aliases.extend(aliases)
        if matched_aliases:
            return _contains_any(ev, matched_aliases)
    # ---------------------------------------------------------
    # TECHNICAL SKILLS
    # ---------------------------------------------------------
    if requirement_type == "technical_skill":
        # -----------------------------------------------------
        # NLP / Transformer / LLM requirements
        # -----------------------------------------------------
        nlp_aliases = [
            "nlp",
            "natural language processing",
            "llm",
            "llms",
            "large language model",
            "large language models",
            "transformer",
            "transformers",
            "sentence-bert",
            "bert",
            "roberta",
            "embedding",
            "embeddings",
            "semantic similarity",
            "information retrieval",
        ]
        if (
            "natural language processing" in req
            or "nlp" in req
            or "transformer" in req
            or "transformers" in req
            or "llm" in req
            or "large language model" in req
        ):
            return _contains_any(ev, nlp_aliases)
        # -----------------------------------------------------
        # ML evaluation requirements
        # -----------------------------------------------------
        evaluation_aliases = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1-score",
            "f1 score",
            "auc",
            "roc",
            "evaluation metrics",
            "model evaluation",
            "evaluate",
            "evaluated",
            "evaluation",
        ]
        if (
            "precision" in req
            or "recall" in req
            or "f1" in req
            or "accuracy" in req
            or "evaluate" in req
            or "evaluation" in req
            or "metrics" in req
        ):
            return _contains_any(ev, evaluation_aliases)
        # -----------------------------------------------------
        # Docker deployment/containerization
        # -----------------------------------------------------
        if "docker" in req:
            deployment_aliases = [
                "dockerized",
                "dockerised",
                "containerized",
                "containerised",
                "containerization",
                "containerisation",
                "docker container",
                "deployed using docker",
                "deployed with docker",
                "deployed docker",
                "docker deployment",
                "dockerized application",
                "dockerised application",
            ]
            return _contains_any(ev, deployment_aliases)
        # -----------------------------------------------------
        # Generic technical skill matching
        # -----------------------------------------------------
        req_tokens = _token_set(req)
        ev_tokens = _token_set(ev)
        return bool(req_tokens & ev_tokens)
    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    req_tokens = _token_set(req)
    ev_tokens = _token_set(ev)
    return bool(req_tokens & ev_tokens)
