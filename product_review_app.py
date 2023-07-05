from homework3 import TextAnalysis
import pprint as pp
import re


def main():
    # initialize framework
    ta = TextAnalysis()

    # register 10 text files (reviews of Twilight)
    results = [ta.load_text('data/twilight_rev1_star3.txt', 'Review 1'),
               ta.load_text('data/twilight_rev2_star5.txt', 'Review 2'),
               # ta.load_text('data/twilight_rev3_star1.txt', 'Review 3'),
               ta.load_text('data/twilight_rev4_star1.txt', 'Review 4'),
               ta.load_text('data/twilight_rev5_star2.txt', 'Review 5'),
               ta.load_text('data/twilight_rev6_star2.txt', 'Review 6'),
               ta.load_text('data/twilight_rev7_star3.txt', 'Review 7'),
               ta.load_text('data/twilight_rev8_star4.txt', 'Review 8'),
               ta.load_text('data/twilight_rev9_star4.txt', 'Review 9'),
               ta.load_text('data/twilight_rev10_star5.txt', 'Review 10')
               # ta.load_text('data/CHANGE_HOMEWORK_NAMES.json')
               ]

    # produce some visualizations
    pp.pprint(results)
    ta.make_sankey(results, 20)
    ta.pie_chart_viz(results, "Sentiment Analysis of Twilight Reviews")
    ta.total_viz(results, "Sentiment and Total Word Correlation")


if __name__ == '__main__':
    main()
