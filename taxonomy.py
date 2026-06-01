"""
Skills taxonomy for engineering job-posting skill extraction.

Four categories form the label scheme used by both the keyword baseline
(src/baseline.py) and the BERT NER tagger (src/model.py):

    TECHNICAL      domain knowledge, methods, theory
    TOOLS          named software, languages, platforms, instruments
    SOFT           transferable / interpersonal skills
    CERT           certifications, licenses, formal credentials

Each entry also carries a discipline hint (me / ee / se / any) used only for
analysis and corpus tagging — it does NOT restrict matching. A "CAD" skill in a
software posting is still a valid match; the hint just records where a skill is
*typically* expected.

Keep surface forms lowercase. Multi-word forms are matched as phrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    TECHNICAL = "TECHNICAL"
    TOOLS = "TOOLS"
    SOFT = "SOFT"
    CERT = "CERT"


# Discipline hints (analysis only — never used to filter matches)
ME = "me"   # mechanical
EE = "ee"   # electrical
SE = "se"   # software
ANY = "any"


@dataclass(frozen=True)
class Skill:
    """A canonical skill plus its surface forms (aliases)."""
    canonical: str
    category: Category
    discipline: str = ANY
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def surface_forms(self) -> list[str]:
        """All lowercase strings that should map to this skill."""
        forms = {self.canonical.lower(), *(a.lower() for a in self.aliases)}
        return sorted(forms, key=len, reverse=True)  # longest-first for matching


# --------------------------------------------------------------------------- #
# The taxonomy. Curated to be representative, not exhaustive — extend freely.
# --------------------------------------------------------------------------- #
SKILLS: list[Skill] = [
    # ---- TECHNICAL: mechanical -------------------------------------------- #
    Skill("Finite Element Analysis", Category.TECHNICAL, ME, ("fea", "finite element")),
    Skill("Computational Fluid Dynamics", Category.TECHNICAL, ME, ("cfd",)),
    Skill("Thermodynamics", Category.TECHNICAL, ME),
    Skill("Heat Transfer", Category.TECHNICAL, ME),
    Skill("GD&T", Category.TECHNICAL, ME,
          ("geometric dimensioning and tolerancing", "geometric dimensioning")),
    Skill("Tolerance Analysis", Category.TECHNICAL, ME, ("tolerance stackup",)),
    Skill("Statics and Dynamics", Category.TECHNICAL, ME, ("statics", "dynamics")),
    Skill("DFM", Category.TECHNICAL, ME, ("design for manufacturability", "design for manufacturing")),
    Skill("Mechanical Design", Category.TECHNICAL, ME),

    # ---- TECHNICAL: electrical -------------------------------------------- #
    Skill("PCB Design", Category.TECHNICAL, EE, ("printed circuit board", "pcb layout")),
    Skill("Signal Processing", Category.TECHNICAL, EE, ("dsp", "digital signal processing")),
    Skill("RF Engineering", Category.TECHNICAL, EE, ("rf", "radio frequency")),
    Skill("Embedded Systems", Category.TECHNICAL, EE, ("embedded",)),
    Skill("Power Electronics", Category.TECHNICAL, EE),
    Skill("Control Systems", Category.TECHNICAL, EE, ("control theory",)),
    Skill("Analog Circuit Design", Category.TECHNICAL, EE, ("analog design",)),
    Skill("Digital Logic Design", Category.TECHNICAL, EE, ("digital design",)),
    Skill("FPGA Development", Category.TECHNICAL, EE, ("fpga",)),

    # ---- TECHNICAL: software ---------------------------------------------- #
    Skill("Machine Learning", Category.TECHNICAL, SE, ("ml",)),
    Skill("Data Structures and Algorithms", Category.TECHNICAL, SE,
          ("data structures", "algorithms")),
    Skill("Distributed Systems", Category.TECHNICAL, SE),
    Skill("REST APIs", Category.TECHNICAL, SE, ("rest api", "restful", "api design")),
    Skill("Microservices", Category.TECHNICAL, SE),
    Skill("Object-Oriented Programming", Category.TECHNICAL, SE, ("oop",)),
    Skill("Database Design", Category.TECHNICAL, SE, ("relational databases",)),
    Skill("Natural Language Processing", Category.TECHNICAL, SE, ("nlp",)),

    # ---- TECHNICAL: cross-discipline -------------------------------------- #
    Skill("CAD", Category.TECHNICAL, ANY, ("computer-aided design", "computer aided design")),
    Skill("Systems Engineering", Category.TECHNICAL, ANY),
    Skill("Root Cause Analysis", Category.TECHNICAL, ANY, ("rca",)),
    Skill("Design of Experiments", Category.TECHNICAL, ANY, ("doe",)),

    # ---- TOOLS: mechanical ------------------------------------------------ #
    Skill("SolidWorks", Category.TOOLS, ME),
    Skill("CATIA", Category.TOOLS, ME),
    Skill("Creo", Category.TOOLS, ME, ("pro/engineer", "pro engineer")),
    Skill("ANSYS", Category.TOOLS, ME),
    Skill("AutoCAD", Category.TOOLS, ME),
    Skill("NX", Category.TOOLS, ME, ("siemens nx", "unigraphics")),

    # ---- TOOLS: electrical ------------------------------------------------ #
    Skill("Altium Designer", Category.TOOLS, EE, ("altium",)),
    Skill("Cadence", Category.TOOLS, EE, ("cadence allegro", "orcad")),
    Skill("LTspice", Category.TOOLS, EE, ("spice",)),
    Skill("Verilog", Category.TOOLS, EE),
    Skill("VHDL", Category.TOOLS, EE),
    Skill("Oscilloscope", Category.TOOLS, EE),
    Skill("MATLAB", Category.TOOLS, ANY),  # heavily used in EE but cross-disc
    Skill("Simulink", Category.TOOLS, ANY),

    # ---- TOOLS: software -------------------------------------------------- #
    Skill("Python", Category.TOOLS, SE),
    Skill("Java", Category.TOOLS, SE),
    Skill("C++", Category.TOOLS, SE, ("cpp",)),
    Skill("JavaScript", Category.TOOLS, SE, ("js",)),
    Skill("SQL", Category.TOOLS, SE),
    Skill("Docker", Category.TOOLS, SE),
    Skill("Kubernetes", Category.TOOLS, SE, ("k8s",)),
    Skill("Git", Category.TOOLS, SE),
    Skill("AWS", Category.TOOLS, SE, ("amazon web services",)),
    Skill("PyTorch", Category.TOOLS, SE),
    Skill("TensorFlow", Category.TOOLS, SE),
    Skill("React", Category.TOOLS, SE),

    # ---- SOFT: cross-discipline ------------------------------------------- #
    Skill("Communication", Category.SOFT, ANY, ("communication skills", "verbal communication")),
    Skill("Teamwork", Category.SOFT, ANY, ("collaboration", "team player")),
    Skill("Problem Solving", Category.SOFT, ANY, ("problem-solving", "analytical thinking")),
    Skill("Leadership", Category.SOFT, ANY),
    Skill("Project Management", Category.SOFT, ANY),
    Skill("Time Management", Category.SOFT, ANY),
    Skill("Attention to Detail", Category.SOFT, ANY),
    Skill("Adaptability", Category.SOFT, ANY, ("flexibility",)),

    # ---- CERT: cross-discipline ------------------------------------------- #
    Skill("PE License", Category.CERT, ANY, ("professional engineer", "p.e.")),
    Skill("FE Exam", Category.CERT, ANY, ("fundamentals of engineering", "eit")),
    Skill("PMP", Category.CERT, ANY, ("project management professional",)),
    Skill("Six Sigma", Category.CERT, ANY, ("lean six sigma", "green belt", "black belt")),
    Skill("Security Clearance", Category.CERT, ANY, ("secret clearance", "ts/sci")),
    Skill("AWS Certified", Category.CERT, SE, ("aws certification",)),
    Skill("CISSP", Category.CERT, ANY),
    Skill("CompTIA Security+", Category.CERT, ANY, ("security+",)),
]


# --------------------------------------------------------------------------- #
# BIO tagging label scheme for token-classification (NER).
# B-/I- prefix per category, plus the O (outside) tag.
# --------------------------------------------------------------------------- #
def _build_labels() -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = ["O"]
    for cat in Category:
        labels.append(f"B-{cat.value}")
        labels.append(f"I-{cat.value}")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label


LABELS, LABEL2ID, ID2LABEL = _build_labels()


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #
def surface_to_skill() -> dict[str, Skill]:
    """Map every lowercase surface form -> its Skill (longest forms win)."""
    mapping: dict[str, Skill] = {}
    for skill in SKILLS:
        for form in skill.surface_forms():
            mapping.setdefault(form, skill)
    return mapping


def skills_by_category(category: Category) -> list[Skill]:
    return [s for s in SKILLS if s.category is category]


def skills_by_discipline(discipline: str) -> list[Skill]:
    return [s for s in SKILLS if s.discipline in (discipline, ANY)]


if __name__ == "__main__":
    print(f"{len(SKILLS)} skills across {len(Category)} categories")
    for cat in Category:
        n = len(skills_by_category(cat))
        print(f"  {cat.value:10s} {n:3d}")
    print(f"{len(LABELS)} NER labels: {LABELS}")
