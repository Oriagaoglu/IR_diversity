"""
Group annotation tool — 4 annotators vote on the same screen.

Usage: python scripts/annotator_app.py
"""

import sys
import json
import time
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QProgressBar, QFrame, QMessageBox, QSizePolicy,
        QGridLayout
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont, QKeySequence, QShortcut
except ImportError:
    print("PyQt6 not installed. Run: pip install PyQt6")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAVE_DIR = PROJECT_ROOT / "data" / "annotations"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATORS = ["orhan", "melani", "constantinos", "zhiyong"]
KEYS_YES = {"1": 0, "2": 1, "3": 2, "4": 3}  # 1-4 = Yes
KEYS_NO = {"Q": 0, "W": 1, "E": 2, "R": 3}   # Q/W/E/R = No


class GroupAnnotatorApp(QWidget):
    def __init__(self):
        super().__init__()

        with open(DATA_DIR / "annotation_100pairs.json") as f:
            data = json.load(f)
        self.pairs = data['pairs']
        self.total = len(self.pairs)

        # Load existing progress per annotator
        self.annotations = {}
        for name in ANNOTATORS:
            path = SAVE_DIR / f"annotator_{name}.json"
            if path.exists():
                with open(path) as f:
                    self.annotations[name] = json.load(f)
            else:
                self.annotations[name] = {}

        self.current = 0
        # Find first pair where not all 4 have voted
        for i, p in enumerate(self.pairs):
            pid = str(p['id'])
            if not all(pid in self.annotations[n] for n in ANNOTATORS):
                self.current = i
                break

        self.votes = {}  # current round: {name: bool}
        self.start_time = time.time()
        self.init_ui()
        self.show_pair()

    def init_ui(self):
        self.setWindowTitle("Group Annotation — 4 Annotators")
        self.setMinimumSize(900, 750)
        self.setStyleSheet("QWidget { background: #1a1a2e; color: #e0e0e0; }")

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 15, 20, 15)

        # Top bar
        top = QHBoxLayout()
        self.progress_label = QLabel()
        self.progress_label.setFont(QFont("Helvetica", 11))
        top.addWidget(self.progress_label)
        top.addStretch()
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Helvetica", 10))
        self.stats_label.setStyleSheet("color: #888;")
        top.addWidget(self.stats_label)
        layout.addLayout(top)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.total)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background: #16213e; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #4ecca3; border-radius: 3px; }
        """)
        layout.addWidget(self.progress_bar)

        # Headline
        self.headline_label = QLabel()
        self.headline_label.setFont(QFont("Helvetica", 9))
        self.headline_label.setStyleSheet(
            "color: #888; padding: 4px 8px; background: #16213e; border-radius: 4px;")
        self.headline_label.setWordWrap(True)
        layout.addWidget(self.headline_label)

        # Question
        self.question_label = QLabel()
        self.question_label.setFont(QFont("Helvetica", 10))
        self.question_label.setStyleSheet("color: #a0a0a0; padding: 2px 8px;")
        self.question_label.setWordWrap(True)
        layout.addWidget(self.question_label)

        # Claim box
        claim_frame = QFrame()
        claim_frame.setStyleSheet(
            "background: #0f3460; border-radius: 8px; padding: 12px;")
        claim_layout = QVBoxLayout(claim_frame)
        claim_header = QLabel("CLAIM")
        claim_header.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        claim_header.setStyleSheet("color: #4ecca3; padding: 0;")
        claim_layout.addWidget(claim_header)
        self.claim_label = QLabel()
        self.claim_label.setFont(QFont("Georgia", 13))
        self.claim_label.setWordWrap(True)
        self.claim_label.setStyleSheet("color: #fff; padding: 0;")
        claim_layout.addWidget(self.claim_label)
        layout.addWidget(claim_frame)

        # Paragraph box
        para_frame = QFrame()
        para_frame.setStyleSheet(
            "background: #16213e; border-radius: 8px; padding: 12px;")
        para_layout = QVBoxLayout(para_frame)
        para_header = QLabel("PARAGRAPH")
        para_header.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        para_header.setStyleSheet("color: #e94560; padding: 0;")
        para_layout.addWidget(para_header)
        self.para_label = QLabel()
        self.para_label.setFont(QFont("Helvetica", 11))
        self.para_label.setWordWrap(True)
        self.para_label.setStyleSheet("color: #ccc; padding: 0;")
        self.para_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        para_layout.addWidget(self.para_label)
        layout.addWidget(para_frame, stretch=1)

        # Prompt
        prompt = QLabel("Does this paragraph express or support the claim?")
        prompt.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setStyleSheet("color: #fff; padding: 4px;")
        layout.addWidget(prompt)

        # Voting grid: 4 columns, each with name + Yes/No buttons
        vote_grid = QGridLayout()
        vote_grid.setSpacing(8)

        colors = ["#e94560", "#4ecca3", "#3498db", "#f39c12"]
        self.vote_btns = {}
        self.vote_indicators = {}

        for col, (name, color) in enumerate(zip(ANNOTATORS, colors)):
            # Name label
            name_label = QLabel(name.capitalize())
            name_label.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet(f"color: {color};")
            vote_grid.addWidget(name_label, 0, col)

            # Vote indicator
            indicator = QLabel("—")
            indicator.setFont(QFont("Helvetica", 18, QFont.Weight.Bold))
            indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            indicator.setFixedHeight(40)
            indicator.setStyleSheet(
                "color: #555; background: #16213e; border-radius: 6px;")
            self.vote_indicators[name] = indicator
            vote_grid.addWidget(indicator, 1, col)

            # Yes button
            btn_yes = QPushButton(f"Yes ({col+1})")
            btn_yes.setFixedHeight(40)
            btn_yes.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            btn_yes.setStyleSheet(f"""
                QPushButton {{ background: {color}; color: #1a1a2e; border: none;
                             border-radius: 6px; }}
                QPushButton:hover {{ background: #fff; }}
            """)
            btn_yes.clicked.connect(lambda _, n=name: self.vote(n, True))
            vote_grid.addWidget(btn_yes, 2, col)

            # No button
            no_keys = list(KEYS_NO.keys())
            btn_no = QPushButton(f"No  ({no_keys[col]})")
            btn_no.setFixedHeight(40)
            btn_no.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            btn_no.setStyleSheet("""
                QPushButton { background: #333; color: #aaa; border: 1px solid #555;
                             border-radius: 6px; }
                QPushButton:hover { background: #555; }
            """)
            btn_no.clicked.connect(lambda _, n=name: self.vote(n, False))
            vote_grid.addWidget(btn_no, 3, col)

        layout.addLayout(vote_grid)

        # Navigation
        nav = QHBoxLayout()
        self.btn_back = QPushButton("< Back")
        self.btn_back.setFixedHeight(34)
        self.btn_back.setStyleSheet("""
            QPushButton { background: #16213e; color: #888; border: 1px solid #333;
                         border-radius: 6px; font-size: 12px; padding: 0 16px; }
            QPushButton:hover { background: #1a2744; }
        """)
        self.btn_back.clicked.connect(self.go_back)
        nav.addWidget(self.btn_back)

        nav.addStretch()

        self.btn_next = QPushButton("Next (Space) >")
        self.btn_next.setFixedHeight(34)
        self.btn_next.setStyleSheet("""
            QPushButton { background: #16213e; color: #888; border: 1px solid #333;
                         border-radius: 6px; font-size: 12px; padding: 0 16px; }
            QPushButton:hover { background: #1a2744; }
        """)
        self.btn_next.clicked.connect(self.go_next)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        self.setLayout(layout)

        # Keyboard shortcuts: 1-4 = Yes, Q/W/E/R = No
        for key, idx in KEYS_YES.items():
            QShortcut(QKeySequence(key), self,
                      lambda n=ANNOTATORS[idx]: self.vote(n, True))
        for key, idx in KEYS_NO.items():
            QShortcut(QKeySequence(key), self,
                      lambda n=ANNOTATORS[idx]: self.vote(n, False))
        QShortcut(QKeySequence("Left"), self, self.go_back)
        QShortcut(QKeySequence("Right"), self, self.go_next)
        QShortcut(QKeySequence("Space"), self, self.go_next)

    def vote(self, annotator, covered):
        pid = str(self.pairs[self.current]['id'])
        self.annotations[annotator][pid] = covered
        self.votes[annotator] = covered
        self.save(annotator)
        self.update_indicator(annotator, covered)

        # Auto-advance when all 4 have voted
        if len(self.votes) == 4 and self.current < self.total - 1:
            self.current += 1
            self.votes = {}
            self.show_pair()

    def update_indicator(self, annotator, covered):
        ind = self.vote_indicators[annotator]
        if covered:
            ind.setText("YES")
            ind.setStyleSheet(
                "color: #4ecca3; background: #1a3a2e; border-radius: 6px; "
                "border: 2px solid #4ecca3;")
        else:
            ind.setText("NO")
            ind.setStyleSheet(
                "color: #e94560; background: #3a1a1e; border-radius: 6px; "
                "border: 2px solid #e94560;")

    def show_pair(self):
        p = self.pairs[self.current]
        pid = str(p['id'])

        self.headline_label.setText(f"Event: {p['headline']}")
        self.question_label.setText(f"Q: {p['question']}")
        self.claim_label.setText(f"\"{p['claim_text']}\"")
        self.para_label.setText(p['paragraph_text'])

        # Count fully annotated pairs
        done = sum(1 for pp in self.pairs
                   if all(str(pp['id']) in self.annotations[n] for n in ANNOTATORS))
        self.progress_bar.setValue(done)
        self.progress_label.setText(
            f"Pair {self.current + 1} / {self.total}  —  {done} complete")

        elapsed = int(time.time() - self.start_time)
        self.stats_label.setText(f"{elapsed // 60}m {elapsed % 60}s")

        # Reset or restore vote indicators
        self.votes = {}
        for name in ANNOTATORS:
            if pid in self.annotations[name]:
                val = self.annotations[name][pid]
                self.votes[name] = val
                self.update_indicator(name, val)
            else:
                ind = self.vote_indicators[name]
                ind.setText("—")
                ind.setStyleSheet(
                    "color: #555; background: #16213e; border-radius: 6px;")

    def go_back(self):
        if self.current > 0:
            self.current -= 1
            self.show_pair()

    def go_next(self):
        if self.current < self.total - 1:
            self.current += 1
            self.show_pair()
        else:
            self.check_done()

    def save(self, annotator):
        path = SAVE_DIR / f"annotator_{annotator}.json"
        with open(path, 'w') as f:
            json.dump(self.annotations[annotator], f, indent=2)

    def check_done(self):
        done = sum(1 for p in self.pairs
                   if all(str(p['id']) in self.annotations[n] for n in ANNOTATORS))
        if done >= self.total:
            QMessageBox.information(
                self, "All Done!",
                f"All {self.total} pairs annotated by all 4 annotators!\n\n"
                f"Run: python scripts/compute_annotation_agreement.py\n"
                f"to compute kappa and validate the LLM judge."
            )


def main():
    app = QApplication(sys.argv)
    window = GroupAnnotatorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
