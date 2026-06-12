from metrics_test import metricsTest
from draw_metrics import draw_from_csv_compare_algorithm, draw_from_csv_compare_reward
from sweep_test import sweepTest, draw_sweep_from_csv


def main():
    metricsTest()
    draw_from_csv_compare_algorithm()
    draw_from_csv_compare_reward()
    sweepTest()
    draw_sweep_from_csv()


if __name__ == "__main__":
    main()
