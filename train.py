import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for live window
import matplotlib.pyplot as plt
import threading
import queue
import time
from agent import Agent
from game import SnakeGameAI


class LivePlotter:
    """
    Live plotter that runs matplotlib in a separate thread.
    Main thread sends data via a queue; this thread renders the plot.
    """

    def __init__(self):
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Thread main loop: create figure and update it on new data."""
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Training Progress (Live)')
        ax.set_xlabel('Number of Games')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)

        line_scores, = ax.plot([], [], label='Score', alpha=0.5, color='steelblue')
        line_mean, = ax.plot([], [], label='Mean Score', linewidth=2, color='darkorange')
        ax.legend(loc='upper left')

        text_score = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        text_record = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)

        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        while not self.stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=0.3)
            except queue.Empty:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
                continue

            scores, mean_scores, record = data

            x = list(range(1, len(scores) + 1))
            line_scores.set_data(x, scores)
            line_mean.set_data(x, mean_scores)

            if len(scores) > 0:
                max_y = max(max(scores), max(mean_scores))
                ax.set_ylim(0, max(5, max_y + 2))

            ax.set_xlim(1, max(10, len(scores)))

            text_score.set_text(f'Latest Score: {scores[-1]}')
            text_record.set_text(f'Record: {record}')

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        plt.close(fig)

    def update(self, scores, mean_scores, record):
        self.data_queue.put((list(scores), list(mean_scores), record))

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


def train():
    """
    Training with LIVE rendering (Pygame + matplotlib) for video demo.
    Optimized for speed: small network, fast epsilon decay, high FPS.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()

    # LIVE rendering (Pygame window visible)
    game = SnakeGameAI(render=True)

    # Start live plotter thread
    plotter = LivePlotter()

    print("=" * 50)
    print("شروع آموزش با نمایش زنده...")
    print("  - پنجره Pygame: مار را در حال یادگیری می‌بینید")
    print("  - پنجره matplotlib: نمودار زنده آموزش")
    print("  - برای توقف Ctrl+C بزنید")
    print("=" * 50)

    start_time = time.time()

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                    print(f'  *** New record! Model saved. ***')

                print(f'Game {agent.n_games}  Score {score}  Record {record}')

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                plotter.update(plot_scores, plot_mean_scores, record)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"آموزش متوقف شد. زمان: {elapsed:.1f} ثانیه")
        print("=" * 50)
        save_plot(plot_scores, plot_mean_scores, agent.n_games)
        plotter.stop()
        print(f"بهترین امتیاز: {record}")
        print(f"تعداد بازی‌ها: {agent.n_games}")
        print(f"میانگین امتیاز: {total_score / max(1, agent.n_games):.1f}")


def save_plot(scores, mean_scores, n_games):
    """Save a final PNG snapshot."""
    plt.figure(figsize=(10, 6))
    plt.title(f'Training Progress (after {n_games} games)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.5, color='steelblue')
    plt.plot(mean_scores, label='Mean Score', linewidth=2, color='darkorange')
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_progress_final.png', dpi=100)
    plt.close()
    print("نمودار نهایی ذخیره شد: training_progress_final.png")


if __name__ == '__main__':
    train()