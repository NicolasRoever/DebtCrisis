from debt_crisis.gpt_sentiment_index.gpt_index_analysis import (
    calculate_gpt_sentiment_index,
)

from debt_crisis.sentiment_index.clean_sentiment_data import preprocess_transcript_text

import pytest
import pandas as pd
import numpy as np


def test_calculate_gpt_sentiment_index():
    # Sample preprocessed data for testing
    data = {
        "Date": pd.date_range(start="2003-01-01", periods=10, freq="D"),
        "Prediction": [1, 2, 2, -1, -2, 2, 2, 1, 1, 0],
    }
    preprocessed_data = pd.DataFrame(data)

    # Define the country under study
    country_under_study = "CountryA"

    # Expected output for the test case
    expected_output = {
        "Date": pd.date_range(start="2003-01-01", periods=10, freq="D"),
        "Sentiment_GPT_CountryA": [
            1,
            1.5,
            1.66667,
            1,
            0.4,
            0.666667,
            0.857143,
            0.875000,
            0.888889,
            0.800000,
        ],
    }
    expected_output_df = pd.DataFrame(expected_output)

    actual_output = calculate_gpt_sentiment_index(
        preprocessed_data, country_under_study, day_window=90
    )

    # Check if the calculated values are as expected
    for idx, row in expected_output_df.iterrows():
        date = row["Date"]
        expected_value = row[f"Sentiment_GPT_{country_under_study}"]
        result_value = actual_output.loc[
            actual_output["Date"] == date, f"Sentiment_GPT_{country_under_study}"
        ].values
        if pd.isna(expected_value):
            assert pd.isna(
                result_value
            ), f"Expected NaN for date {date}, but got {result_value}"
        else:
            assert np.isclose(
                result_value, expected_value, atol=0.01
            ), f"Sentiment index for {date} does not match. Expected: {expected_value}, Got: {result_value}"

    # @pytest.fixture
    # def test_transcript():
    return r"""
Thomson Reuters StreetEvents Event Transcript
E D I T E D   V E R S I O N

2003 Baloise Holding Earnings Analyst Meeting &amp; Conference Call
APRIL 06, 2004 / 11:00AM GMT

================================================================================
Corporate Participants
================================================================================

 * Christian Hippemeyer
   Baloise Holding - Chief Actuary
 * Wolfgang Drunk
   Baloise Holding - CFO
 * Martin Strobel
   Baloise Holding - CEO

================================================================================
Conference Call Participiants
================================================================================

 * Stefan Schurman
   Pictet & Cie. - Analyst
 * Peter Casanova
   Deutsche Bank - Analyst
 * Thomas Schwarzenbach
   Bank am Bellevue - Analyst
 * Heine Wimmer
   Bank Oppenheim - Analyst
 * Duncan Russell
   SBK - Analyst
 * Laurent Rousseau
   Credit Suisse First Boston - Analyst

================================================================================
Presentation
--------------------------------------------------------------------------------
Operator    [1]
--------------------------------------------------------------------------------
Welcome and thank you for joining the Baloise Insurance conference call. (OPERATOR INSTRUCTIONS) We kindly ask you to limit yourself to one question only. At this time, you will be joined into the conference room with Mr. Wolfgang Drunk, CFO of Baloise. You will be now joined to the conference. Thank you.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [2]
--------------------------------------------------------------------------------
Ladies and gentlemen, I would like to welcome you to the presentation of the results of the 2003 business year of the Baloise Group. And I would like to in particular welcome those who are on the Internet and those who are listening to us via telephone call, so they are going to be with us for the first time on line this year and even put questions to us later on by conference call. The camera is at the back of the hall on the left for the webcast, and we have a technician and the crew that are upstairs behind the glass window and are there to help us.
The reason why I as Chairman of the Board am going to conduct the proceedings here instead of Frank Schnewlin is very simple. Mr. Frank Schnewlin, who is a native of the Bernese Oberland and a very good skier therefore, broke his sternum just a few days ago when he was skiing. Mr. Schnewlin will be leaving the hospital this week, but to have an impeccable recovery, he needs a little bit of rest. This afternoon he will address the management meeting of Baloise and will in a very short time return to work.
Frank Schnewlin has asked me to convey his best greetings to you, and I am quite sure that I can on your behalf convey to him our best wishes for a speedy recovery. Now, all of the persons present in this group are requested to turn off their mobiles, otherwise there will be interference with the WebCams.
I will now present to you the key developments of 2003. Martin Strobel will then inform you about the developments in progress in our Swiss business. I will then conclude with a few remarks about the outlook for the next year, 2004; and the entire group management will then be at your disposal for questions. So, we will first of all take questions from the room here, from the hall here, and then questions from those listening to us on the Internet and via conference call.
The nutshell of my message to you is as follows. By substantially improving its operational earning power in 2003, the Baloise is strongly positioned for future development. Let me start out by reviewing the key developments of the financial year 2003. The Baloise Group has substantially improved its operational performance, realizing a net profit of 91 million Swiss francs, compared to a loss of 634 million francs in the previous year. All lines of business contributed to this achievement.
The result reflects the success of the numerous measures implemented across the board with a view to increasing operational earning power. By exceeding market expectations, this result demonstrates that we're moving in the right direction. Thus we have an important stage behind us, but we have not reached our goal yet. In 2004, we shall continue to improve our business portfolio and processes so as to further enhance our operating earnings power distinctly and therefore create the prerequisite for further profitable growth.
We will ask the Annual General Meeting on the 14th of May 2004 to approve a cash dividend of 60 centimes per share; that will be an increase of 50 percent compared with 2002. The Baloise cultivates a distribution policy of regular profit-linked dividend payments tailored to the needs of its long-term investors.
A year ago we presented to you a package of detailed measures to turn around our business portfolios in various markets. I am very pleased to report to you today that we have successfully completed our efforts to restructure our underwriting business. Nonetheless, we shall continue to optimize all our portfolios with the view, as mentioned previously, of improving operational earning power. I shall deal separately with the measures adopted in the cases of Belgium and Baseler Securitas.
The effects of these restructuring measures are impressive. In non-life, the gross combined ratio improved by 17.6 percentage points to 97.6. In the life business, the embedded value rose sharply by 21.4 percent. We're particularly pleased with the overall performance of our capital investments, with an increase in value by 4.6 percent; at the end of 2002, the corresponding figure was still minus 0.9 percent. This gratifying turnaround is the consequence of the upbeat stock market sentiment and positive currency effects.
Apart from the improvement in earning power, another major goal was the straightening of our net asset value. Thanks to our active risk-aligned investment management, consolidated group solvency rose to 241 percent compared with 231 percent in the previous year. We reduced the equity quota to 6.2 percent and at a favorable time shifted a substantial portion of the fixed interest-bearing securities to the held-to-maturity category.
The capital and reserves of the Baloise also developed satisfactorily, rising by 7.5 percent year-on-year to 3.3 billion Swiss francs. The principal factors influencing this increase were net profit, positive growth in the value of our capital investments, and currency gains. The Baloise has also proven that it takes its social and environmental responsibility seriously. In the recently published sustainability rating produced by Sustainable Investment Research International, it ranks seventh out of the 25 primary insurers included in the SMI.
The premium income rose to 7.4 billion Swiss francs from 7.3 billion in 2002, which represents a growth in Swiss francs of 1.4 percent. This figure is a precise reflection of the Baloise's current business policies which are based on the following principles. First of all, earnings before growth; secondly, new business must be profitable; thirdly, non-life is the focus of our business expansion.
Thanks to the acquisition of Securitas, the development in premium volumes since the end of 2002 also includes welcome diversification effects. By region, Germany's share has risen to 30 percent at the expense of Switzerland, whose quota has fallen from 64 to 58 percent. Moreover, there has also been a shift by business. Non-life now accounts for 42 percent; in 2002, it was 36 percent. And life, 58 percent; it was 64 percent in 2002. So, as you can see all divisions have contributed to our profits.
In my second presentation on financial aspects, I will be giving you more detail of tax on income. The non-life business produced profits before tax and minority interest of 91 Swiss francs, compared to a loss of 203 million Swiss francs in 2002. This distinct success is the result of consistent portfolio reorganizations, strict cost management, and the absence of major losses.
Premium income rose to 3.1 billion Swiss francs, compared to 2.6 billion in the previous year. This is the equivalent of an increase of 14 percent in local currencies. Organic growth in local currencies amounted to 3 percent. The remaining growth is the result of the acquisition of the Securitas group, which has been consolidated for the first time in the 2003 financial statements. About three-quarters of the Securitas portfolio comprises non-life business; hence the acquisition is consistent with the Baloise's strategic focus on non-life in future business expansion.
There was a very marked improvement in the gross combined ratio, a cost-loss ratio, of 7.6 percentage points to 97.6. The improvement in the net combined ratio value of 7.7 percentage points is equally impressive. Success in portfolio realignments in Germany and Belgium in particular is reflected in a gross loss ratio of 63.7 percent, an improvement of 7.5 percent year-on-year. After taking into account special effects such as provisions for restructuring, the cost ratio improved to 29.3 percent. I will come back to this as well in more detail in the second part of my presentation.
Particularly noticeable are the improvements of Mercator in Belgium, where the gross combined ratio has fallen to 98.1 percent; this was 114.4 percent in 2002. And of Basler Securitas with a gross combined ratio of 99.9 percent, where in fact the most restructuring was called for. Basler Securitas had a gross combined ratio in 2002 of 127 percent.
The life business produced profits before tax and minority interest of 69 million Swiss francs, compared to a loss of 359 million in 2002. Owing to extraordinary tax charges incurred by the German life insurance companies, the net income for 2003 is a more meaningful figure this year; that is to say 27 million Swiss francs. This result demonstrates that the Baloise has successfully structured its life business to changes in general conditions. Crucial measures adopted, including adjusting surpluses to the situation in the financial markets, risk-aligned prices, and strictly profit-oriented underwriting policies. As of the end of 2003, the embedded value had risen by 21.4 percent to 1.98 billion Swiss francs.
The value of the insurance portfolio also posted a strong rise amounted to 1.24 billion Swiss francs. That is an increase of 45 percent since the beginning of 2003. The value of new life business amounted to 15.5 million. The earnings margin is positive in all markets. We have succeeded in making it profitable again, although this division was badly hit by developments in the financial markets. Martin Strobel will discuss the probability of the Swiss life business, in particular group life insurance.
And at this point, I would like to emphasize that the recent decision, the legal quote, that is to say, an unsatisfactory procedure which is difficult to understand, has been adopted. This reduces the transparency and the manageability of life business distinctly and may impair the future profitability or earning power.
Premium income in the life business shrank by 1.8 percent in local currencies because in the interest of preserving its earning power the Baloise only underwrote new life policies that complied with its profitability criteria. The decline in surpluses resulted in lower premium income on the individual life side, most notably in single premium policies.
The banking business produced a profit before tax and minority interest of 37 million Swiss francs, compared to a loss of 100 million in 2002. The Baloise Bank SoBa accounts for the largest part of this positive result. Thanks to changes in its core business, its profit rose by 24 percent. The Mercator Bank in Belgium also contributed to profit. The Deutscher Ring Bausparkasse continued to make progress in keeping with expectations; and if everything goes according to plan should reach profitability by the end of 2006.
Belgium. A year ago, we explained to you the main thrust of our reorganization of Mercator in Belgium, to turn around the insurance and banking businesses and to restructure the investment portfolio. In 2003, we were able to make sizable progress. The technical restructuring of the non-life business has been completed, as documented by the outstanding improvement in the gross combined ratio of 16.3 percentage points to 98.1. The loss ratio also improved by 15 percentage points, rising to 67.7 percent. In the life business we adjusted commitments to the achievable returns on investments. New life business shows a positive earnings margin.
At the Mercator Bank, the measures introduced in 2003 have led to a marked improvement in the earnings situation. Thanks to the realignment of the capital investments and to restructuring effects, profits have improved. In Belgium, we have substantially strengthened management. Both the insurance and banking businesses are being led by new CEOs. The new management is well underway in preparing the ground for future developments.
The immediate focus is on a comprehensive analysis and realignment of all business areas with respect to the sustainable earning power. To this end all strategic options are being examined in view of optimizing the local business model, including those for the Mercator Bank. We have also shifted substantial elements of the investment portfolio, with a view to improving current returns. We have continued to dispose of non-strategic participations.
Altogether, these measures have substantially reduced losses. However, we have not yet, quite obviously, reached our goal and will continue to consistently pursue optimization. In 2004, we will make further significant progress in enhancing Mercator's earning power.
Now to Basler Securitas. In April 2003, we explained to you our priorities for Basler Securitas, namely, the successful merging of the Baloise and Securitas, as well as the technical restructuring of the non-life portfolios to focus on motor and commercial insurance.
Here too we have made substantial progress in 2003. We have completed the restructuring of the insurance portfolio. This has had a marked effect on the gross combined ratio in non-life, which recorded an outstanding improvement of 27.1 percentage, rising to 99.9 percent. In industrial insurance, we have halved the underwriting limits for major risks.
The integration of Securitas and Basler has made solid progress. We concluded the acquisition at a highly attractive price. A new CEO has strengthened the company's management, and the new Basler Securitas brand was launched in October 2003. The operational integration of the management and distribution structures has been completed and the product portfolio by and large harmonized. We expect cost synergy from the integration to manifest itself from 2004 onwards.
This brings me to the end of the first part, for which I was standing in for Mr. Frank Schnewlin, our CFO. And I now come to the second part. Mr. Strobel will be talking about the Swiss Basler and then go on to the main financial aspects. And we will now make my presentation, which was always planned as being presented by myself.
In conjunction with greater stability in capital markets, the implementation of comprehensive measures to improve operations at virtually all areas of our company enabled us to report a consolidated result of 91 million Swiss francs. This signals Baloise's marked return to the profit zone. It gives me great pleasure to explain to you the major financial developments behind this result.
In 2003 investment income developed satisfactorily, amounting to almost 2 billion Swiss francs. Capital and reserves on an IAS basis rose by 232 million or 7.5 percent to about 3.3 billion Swiss francs, which corresponds to a solvency rate on an IAS basis of 241 percent, and capital and reserves per share of 60 Swiss francs.
The combined ratio reflects the underwriting performance, with a gross value of 97.6 percent. This measure fell well below 100 percent. Thus, Baloise fulfilled one of its targets for 2004 a year ahead of time. The net combined ratio also improved by 7.7 percentage points. At this point I should like to add that our reserve ratio, i.e. the technical reserve level adjusted for acquisitions, increased from 181 percent to 186. That is 5 percentage points indeed.
Let me summarize the key factors of Baloise's annual results. In underwriting, these are the marked improvements in operational performance in all segments. This is manifested in a combined ratio 97.6 percent, composed of a loss ratio that improved sharply from 74.8 percent to 67.3 percent, and a cost ratio that edged down from 30.0 percent to 29.9. Included in this cost ratio, however, are special costs such as provisions for restructuring Basler Securitas. Adjusted for these special effects, the cost ratio would be 29.3 percent; and the combined ratio as a result of that would be 97.0 percent.
The full effects of some measures implemented in 2003 will only be properly felt in 2004. Accordingly, we expect another solid increase in earnings power in the current year. At the same time, it should be pointed out that there were very few major claims in 2003, which naturally also contributed to the successful results.
We have already mentioned growth. In non-life it amounted to 14 percent in local currencies and 3 percent adjusted for acquisitions. In the life business premium volume fell by 8 percent in local currencies and 11 percent in adjusted terms for acquisitions. On the other hand, the total premium income of 4.3 billion Swiss francs in life is another very good result. The embedded value of new policies concluded in 2003 amounted to 15.5 million Swiss francs. The embedded value of our portfolio rose by 21 percent from 1,631 million Swiss francs to 1,980 million.
Thanks to changes in the overall environment and adjustments to rates, interest margins were again positive in all segments except for parts of the traditional Belgian and Swiss portfolios, even as interest margins in Switzerland continued to be very thin. Hence, the topic of the minimum rate in the occupational pensions business has not been resolved, unfortunately.
On our capital investments, we were able to book 143 million in currency gains on bonds. This is primarily the result of the positive development in the euro. Thanks to our efficient hedging strategy, we were able to counter the effects of the weak dollar. Equity markets recovered well in the second half of the year. However, owing to our cautious realization policy in the same period, improvements in value are manifested for the most part as unrealized gains. The unrealized gains also include the effects of the modest rise in interest rates. I shall return to these latter two points further down.
On the subject of asset allocation, we were able to invest a large part of our liquid assets in bonds in the phase during which interest rates were recovering. Note, interest rates on 10-year Swiss franc paper is in excess of 2.6 percent; so in this phase we were able to invest a large part of our liquid assets in bonds. The strategy of avoiding large investments in fixed-interest securities at very low interest rates has paid off.
This overdue you're seeing here, and you're already familiar with from the semester presentation, quantifies the exceptional factors affecting capital investments. You can see the negative effects in the first 6 months of 2003 and the solid recovery in the second semester of the year. 202 million Swiss francs in exceptional negative effects in capital investments in the first semester of the year are compared with 161 million Swiss francs in positive effects in the second semester of the year, resulting in an overall loss of 41 million Swiss francs on exceptional items.
The two largest items in the second semester of the year were the disposal of Deutscher Ring shares in the DEPFA Bank, that was an amount of 40 million Swiss francs; and the disposal of the Rubens (ph) real estate from Mercator's at 24 million Swiss francs.
Over and above this, the following exceptional items are to be mentioned at this point. 59 million Swiss francs in tax elements unrelated to the accounting period, of which 38 million Swiss francs are due to changes in tax laws applicable to German life insurers. Furthermore, we have got exceptional depreciation; 42 million Swiss francs were written off in the deferred acquisition cost, and we have exceptional goodwill write-downs of 37 million Swiss francs. I should like to emphasize that after these goodwill write-downs there is in net terms no goodwill remaining on our balance sheet.
By way of an exception, I am today presenting a post-tax segment overview. The reason for this is the aforementioned special tax effects from German life insurance which distort pre-tax comparisons. As always, you will find both sets of figures in the annual report, pre- and post-tax. There's been a considerable improvement in results in all three segments. After-tax, non-life business contributed 48 million Swiss francs to the annual results, whereas the life segment contributed 27 million Swiss francs, and the banking segment contributed 22 million Swiss francs.
This overview summarizes how we manage profit-oriented growth unit by unit. In Switzerland, life premium volume has declined since the Baloise adjusted policy conditions. In non-life, we were able to increase business by 5.3 percent. Developments in Germany were dominated by the process of integrating Securitas, which has considerably strengthened Baloise's position in the strategically important non-life business. Technical restructuring dampened growth in Belgium. However as the next transparency shows, the combined ratio has improved significantly. Let me also draw your attention to the extremely strong growth in the profitable non-life business in Austria, a result in particular of expanding the salesforce while simultaneously cutting costs.
This overview -- you're familiar with -- summarizes the underwriting performance of our core units. As you can see, the Austrian business is not only growing exceptionally strongly, but the loss ratio of 65.6 percent is very good too; especially by comparison with others in the Austrian market. The comparatively high cost ratio is a consequence of small size of the portfolios. The increasingly profitable growth will lead to gradual decline in this cost ratio.
The top performer is our largest unit, Switzerland, with an outstanding cost ratio of 23.8 percent and an excellent combined ratio of 95.9 percent. Belgium, which still at a combined ratio of about 120 percent in 2002, recorded a combined ratio of 98.1 percent this year, making it the strongest second unit in operational terms.
Whereas in 2002 Basler Securitas, or DFD as it was, reported a combined ratio of 127 percent in the merger year, it could already reduce the combined ratio to below 100 percent. Luxembourg is just above this threshold; and Deutscher Ring produced a combined ratio of 103.1 percent.
This is a multiyear overview in which you can see how Baloise has continuously and consistently reduced costs. The difference between 2002 and 2003 is small. The cost ratio includes, as mentioned before however, nonrecurrent effects such as provisions for restructuring; 0.6 percentage point, that is.
Now, we turn to the embedded value in the life business. The embedded value increased by 21 percent to 1.980 billion Swiss francs, to which the value of the insurance portfolio of 1.236 billion, or an increase by 45 percent, made a crucial contribution. The value of new business concluded in 2003 amounted to 15.5 million, which represents a return on embedded value, an internal rate of return of risk discount rate, plus 1.6 percent. You know that we have a target of RDR plus 1 percent. These figures already take into account the planned introduction in 2004, this year, of a legal quote in group life insurance in Switzerland in accordance with the Federal Office of Private Insurance proposal of last December.
The most important factors of change in embedded value are the following. The expected value added, that is, the increase in value of the life portfolio by the risk discount rate on an annual basis, amounted to 124 million Swiss francs. Implementation of the planned legal quote for group life insurance in Switzerland reduced the embedded value by 303 million Swiss francs. Adjustments to the rates for occupational insurance, group life business increased the embedded value by 257 million Swiss francs. The expected lower income from capital investments reduced the embedded value by 197 million Swiss francs. And the reduction in policyholder bonuses contributed 336 million Swiss francs to the embedded value.
Last but not least, developments in the capital markets, positive developments in the capital markets in 2003 added 227 million Swiss francs. In total, on aggregate, the embedded value rose from 1.360 billion by 350 million to just under 2 billion.
Financial results. I have already mentioned that the financial results amounted to 1.988 billion. This figure breaks down into about 2.1 billion Swiss francs in current investment income, a loss of 41 million Swiss francs in realized gains and losses, and 76 (ph) million Swiss francs in costs for investment management.
I should now like to turn to asset allocation. But rather than discuss it on the basis of this transparency that relates to the entire group, I should prefer to use the following transparencies that refer to what we call the insurance companies of our group, with the three banks, Baloise Bank SoBa, Mercator Bank, and Deutscher Ring Bausparkasse having been excluded in asset allocation.
You will also notice that we considerably reduced the equity share, and we have a large liquidity cushion of 13 percent up to mid-2003. You will find that in the footnotes as many other important things. This 13 percent of liquidity we were able to invest in fixed-interest securities at a good interest rate. Hence, fixed-interest securities now account for 55 percent of capital investments. Mortgages were 11 percent; property 12 percent; policy and other loans 3 percent; and private equity and hedge funds together, as alternative financial investments, about another 3 percent.
Gross unrealized gains, going back to the former issue, you get gross unrealized gains, rose by 413 million Swiss francs and amounted to a little less than 1.2 billion Swiss francs at the end of 2003. As I mentioned, the positive development in the stock markets is largely responsible for this; whereas the slight firming in interest rates led to a small decrease in unrealized gains in this area.
As you may recall, a rise in interest rates is positive for an insurance company in economic terms, because duration in the liabilities section of the balance sheet is greater than on the assets side. Under ISRS rules the liabilities side uses nominal values, which explains why in the event of an interest rate rise unrealized gains on the assets side fall if there is no balancing entry; and thus have a negative effect on capital and reserves, again under ISRS rules. This is against economically appropriate facts.
Net unrealized gains and losses, after the deducting deferred taxes, minority interest, shares of policyholders, and foreign exchange differences, rose by 146 million Swiss francs to attain 161 million Swiss francs. These 146 million Swiss francs, together with the annual income, and the 17 million from the disposal of owned shares, less dividends paid out, represent the increase in assets which strengthened IAS-based capital and reserves by 7.5 percent to 3.3 billion Swiss francs.
Concerning the group of shareholders, I would like to point out that since November 2003, all Baloise shares have again been publicly dispersed (ph) and had a free float of 100 percent. We are particularly gratified to note that in the past year the number of shareholders again recorded a substantial increase.
This brings me to talking about the share price, which is less gratifying. As you know, the question of the share price is always one of the period under review. For this reason, we have got three periods that we present to you. In the first seven-year period, the performance of the sharp (ph) Baloise shares compares well to those of its peers. However, the Baloise's gain of 17 percent still underperformed the SMI by 22 percentage points.
Baloise shares did not have a good year in 2003. For the year as a whole, Baloise shares fell by 6 percent and underperformed the index by 25 percentage points. Thus they were at the lower end of the peer group range.
Last but not least, the year-to-date up to April 5, yesterday, Baloise at 10.4 percent had a positive performance and it clearly outperformed the index. The SMI, over the same period of time, 2004, gained 5.9 percent. This was a poor performance especially in 2003, and we are of course extremely dissatisfied with it. We will do everything in our power to implement measures to improve operational earnings power and are convinced that the effects will also be reflected in the share price.
Let's turn to the proposals to be submitted to the Annual General Meeting. We will ask the Annual General Meeting on May 14, 2004, to approve a dividend of 60 centimes per share, which represents an increase of 50 percent, the equivalent of about one-third of Baloise's profits and a dividend yield of 1.2 percent based on the year-end share price.
Apart from that, we will propose to the Annual General Meeting that it authorize a conditional capital increase of 10 percent so as to increase our refinancing flexibility. Upon this, I should like to conclude my remarks, my actual presentation. I would like to hand over to Martin Strobel our Swiss CEO; and we will come back later with prospects for 2004 and for the medium term.

--------------------------------------------------------------------------------
Martin Strobel,  Baloise Holding - CEO    [3]
--------------------------------------------------------------------------------
Thank you very much. Ladies and gentlemen, 2003 was marked by the strict implementation of measures to enhance profitability in our Swiss operations. Both on the insurance and banking side we made significant progress in all areas of operations. This is clearly reflected in the key figures and our results. I would like to now outline the 2003 business results and the impact of the measures in greater detail.
The consistent implementation of our profitability-oriented approach is reflected in the development of the premium income. Non-life registered a pleasing 5.3 percent to 1.238 billion Swiss francs. But of this 5.3 percent about 2.2 percent resulted from actuarially-necessary premium adjustments and portfolio restructuring in existing business. Roughly 3.1 percent comes from new business.
The firm focus on profitability in the life sector and ensuing consolidation after the vigorous growth of the previous year led to a decline in premium income by 12.8 percent to about 3.056 billion Swiss francs. Premium income in Swiss business overall, including investment-type premiums, decreased against the previous year by 8.3 percent to a level of 4.294 billion Swiss francs.
Let me turn to life business. In individual life insurance, there was a 27 percent reduction in premium income versus the year 2002. Lowering the technical interest rate to 2 percent and the profitability-related adjustment of policyholder bonuses to the capital market situation led to the anticipated consolidation comparison with the strong growth one year before. Nevertheless, thanks to the strength of the Baloise salesforce and the cooperation with UBS, the decline in premium intake was held in check, and the market position in a shrinking overall market was expanded further.
Group insurance. Occupational pension schemes in particular registered a slight decrease in premium income by 1.6 percent due primarily to the economy-related decline in single premiums. If we take annual premiums on the other hand, there was a roughly 6 percent advance against the previous year. This is a result of the lower number of contract cancellations and the selective conclusion of new business in our targeted SME customer segment. Higher premium intake was also generated from the existing customer portfolio owing to the annual portfolio adjustment; for example follow-on insurance.
Non-life. Premiums advanced by about 7.9 percent in the individual customer segment. Motor insurance in particular registered a strong increase despite shrinking car sales and our consistent restructuring measures in the fleet business. This was achieved thanks to the strong production figures of our salesforce and the rate adjustments introduced on January 1, 2003, as well as the distribution partnership with TCS, the Touring Club Switzerland. The latter is a fine example of our efficient multi-channel management.
In our combined property and liability sector, attractive products and the requisite rate adjustments led to a clear increase in premium income.
Commercial and corporate customer business recorded a 2.6 percent premium rise, with the property and liability line realizing distinct profit growth in a tough market environment. A decline in premium income on the other hand was experienced by daily allowance insurance in the group health business, due to technically called-for restructuring measures; and by transport insurance, owing to currency effects and restructuring measures.
The combined ratio, that is to say, representing the ratio of losses incurred and cost of premiums earned, was brought down to 95.9 percent, a substantial reduction against the previous year. The reduction was primarily a result of the enhanced operational efficiency achieved by further optimization of processes and structures. As a consequence, the cost ratio sank to 23.8 percent. Baloise's substantially improved efficiency clearly underscores its claim to cost leadership in the Swiss market.
Overall losses incurred were made on the previous year's levels. Motor, accident, and group daily sickness allowance insurance recorded a rise in losses incurred, with the ballooning medical and rehabilitation cost and the increase in the number of road-accident victims driving up claim payment. The main reason for high loss ratio in group daily sickness allowance is a trend to longer periods of disability.
It is against this backdrop that we made the appropriate rate adjustments in 2003 and reinforced our restructuring measures. These measures have proved successful and, in conjunction with a favorable year 2003 as far as natural disasters are concerned, led to a stabilization of the loss ratio.
Baloise Bank SoBa managed to boot its net profit by a highly gratifying 23.8 percent to 12.5 million compared with last year, thanks to improved operational earning power. Baloise Bank SoBa generated a gross profit of 35.6 million Swiss francs in 2003, equivalent to a 7.9 percent increase on the year before.
Income from interest operations decreased by 3.5 percent due to the extremely low interest rate levels an the related cost of hedging interest rate risks. Meanwhile, income from commission and services activities rose by around 9.7 percent to 19.5 million Swiss francs.
Significantly enhanced efficiency and strict risk management with a correspondingly reduced call for value adjustments contributed considerably to the successful result. The cost to income ratio was also improved against the previous year's by nearly 3 percentage points, down to 68.5 percent.
Mobile banking, that is say the distribution of banking products via the insurance salesforce, showed a very pleasing picture. In 2003 our salesforce brokered banking products with a volume of around 205 million Swiss francs. That is 54 percent above the planned target.
Enlarging on this business model, two new initiatives were launched in 2003. The first one is Baloise Hypo-Plus (ph) that was launched in the autumn, which builds on our combined insurance and banking strength in connection with home ownership. The mortgage from Baloise Bank SoBa is linked with household insurance or a policy covering mortgage repayment to form a customized and comprehensive solution. This bundled product is distributed through the Baloise Insurance salesforce and the Bank SoBa outlets. With sales to the tune of about 106 million Swiss francs, the product has exceeded its target by roughly 50 percent.
The second new product, Baloise Life-Plus (ph), combines our risk insurance competence with a transparent savings module from Baloise Bank SoBa to an attractive entity. This product has also met with positive market acclaim and has exceeded our expectations.
Tomorrow, the Baloise Bank SoBa will be able to report about this at its annual media conference in detail. You will find the relevant media information with the documentation provided to you this morning.
Outlook for 2004. The earning power of our Swiss operations was distinctly enhanced by the measures implemented in 2003. We intend to further reinforce it on this basis. We are at present optimizing our restructuring claims segment processes to raise the profitability of our products and reduce the loss ratio.
Two approaches are taken to increase the earning power of the salesforce, again with the view of insuring profitable growth. On the one hand, specific targeting of sales efforts and support measures are intended to raise the salesforce productivity. And on the other hand, we are optimizing the focus on value of the brokered channel. The quality of the generated business has become a prime management indicating our cooperation with brokers.
Lastly, we are continuing to sharpen our customer focus. In our profitable clearly-defined customer segments, we aim to achieve above-average growth by taking targeted measures to strengthen cross-selling and customer loyalty. In the interest of our insured clientele as a whole, we are bringing continuously nonprofitable contracts to a close.
In combination we expect these measures to bring about further improvements in earning power. On the non-life side, we are counting on premium income to pursue its strong growth and we aim to substantially reduce the combined ratio. In the life sector, we anticipate lower premium income against 2003, in line with current market developments, together with an ongoing improvement in profitability. As far as Baloise Bank SoBa is concerned we expect our focused financial service provider business model to generate a further increase in sales. The bank's operational earning power will continue to be enhanced.
Naturally, any unexpected financial market developments or major loss events would impact the outlook given here for 2004. Following this forecast for 2004, I would like to conclude with a few comments on the Baloise's occupational pensions business, or the BVG, as the acronym is known.
It has become a tradition of the Baloise to disclose the operating account of its BVG business in a transparent manner. The BVG results presented here show that over 90 percent of the earnings were distributed to our customers in 2003, as it has in every year since 1985. To further increase transparency and ease of understanding for our customers, we are publishing the statutory financial statement of our Swiss life business; and it has been enclosed in your documentation.
What measures are we adopting? Well, we stand by the BGV and are consistently working to keep our business model attuned to market conditions. An important focal point is raising transparency and efficiency for the benefit of our customers. As of January 1, 2005, we will be offering so-called light products, which thanks to considerably reduced complexity will lead to lower administrative costs.
As of May 2004, our new IT administration system will be put into operation, which should enable us to boost the efficiency of the business processes. As of January 1, 2005, we will be splitting new BVG business. That is to say, that in the extra-mandatory part, the minimum interest rate and the conversion rate will be fixed according to realistic parameters. The existing portfolio will be brought into line with a step-by-step procedure up to the first of January 2006, (inaudible) in line with the (technical difficulty) models.
Without realistic parameters it is not possible for us to conduct BVG business in a sustainable way. The minimum BVG interest rate, a market value, should not be subject to a political decision but should be determined by relevant market factors in a way that is readily understood by all market players. This is the only way that the BVG business can be built on a solid and sustainable foundation. Thus for the sake of clarity and transparency, we advocate the following formula. That the minimum interest rate be defined as a 60 percent of the 10-year rolling interest rate on Swiss Confederation bonds.
Thank you for attention. I can return the floor now to Wolfgang Drunk.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [4]
--------------------------------------------------------------------------------
Now, onto the prospects, the outlook; but not only for the Swiss business but for the entire group. In 2003, we made significant operational progress in all markets. But this is simply a stage in our way to becoming one of the most profitable insurance providers.
The first three months of 2004 have been largely in line with our expectations. The capital markets are developing favorably, and we have not registered any extraordinary claims to date. Premium income in individual life insurance in Switzerland is still low-key, owing to the difficult market circumstances. Our principal focus remains the further improvement in operating results in all markets.
In the non-life business, our goal for 2004 is a gross combined ratio of 97 percent, which is a challenging goal in the light of the low level of major claims in 2003. In view of the further optimization of the business quality in non-life, we expect that organic premium growth will be in line with the market average.
In the life business, we expect the business volume to recede modestly as the framework (ph) conditions are improving only hesitantly. At the same time, our IAS-based results should record a further increase this year.
We remain cautiously optimistic about developments in the financial markets. Nonetheless, we hope to be able to report a marked advance in the net income for 2004.
For us, earnings power is a clearly defined and measurable figure. We want a return on capital and reserves of no less than 10 percent. We have set the goal of above-average premium growth in profitable segments. In the non-life business, our goal is to achieve a combined ratio for 2007 that sustainably outperforms the market. On this, I would like to conclude our presentations and open the floor for questions and answers.

--------------------------------------------------------------------------------
Operator    [5]
--------------------------------------------------------------------------------
(OPERATOR INSTRUCTIONS) We're going to take questions from the hall here first; and after that we're going to take questions from the telephone conference. Who would like to take the floor please? Mr. Dallen (ph).

--------------------------------------------------------------------------------
Unidentified Audience Member    [6]
--------------------------------------------------------------------------------
(technical difficulty) I have a question on banking. The banking segment. You've got a total income of 22 million SoBa 12.5, actually Mercator was strongly positive, and Deutscher Bausparkasse apparently was negative. Why does it take so for Deutscher Bausparkasse? What is the conditions there, Mercator income, as opposed to the loss run up by Bausparkasse?

--------------------------------------------------------------------------------
Unidentified Audience Member    [7]
--------------------------------------------------------------------------------
You're absolutely right in your statements. That is absolutely correct. Deutscher Ring Bausparkasse posted a slight loss; I am looking it up; 6 million Swiss francs is it? About 6 million Swiss francs, that is the loss run up by Deutscher Ring Bausparkasse.
Deutscher Ring Bausparkasse was started two to three years ago -- started restructuring two to three years ago and the program is running very well. We've got considerable new business coming in. The business plan foresees breakeven to be by 2006; and we also planned the 6 million of loss. But apart from the positive results by Baloise Bank SoBa and Mercator Bank, both of them being positive, we've got Baloise Asset Management. That is the Martin Wenk's business. They also of course contributed positively to the result.

--------------------------------------------------------------------------------
Heine Wimmer,  Bank Oppenheim - Analyst    [8]
--------------------------------------------------------------------------------
Heine Wimmer (ph), Bank Oppenheim. Question on the conditional capital. If I understand properly, there is no specific reason or no acquisition being planned; so it is merely a question of refinancing the holding debt, the outstanding holding debts, for you to keep the option open. And I would like to emphasize, to have that coupon favorable, in coupon-favorable terms.
I wanted to be ironical referring to the very nice transparency 15 in the in annual report, where it is about payout for the year 2004. That would be a negative contribution, because you always added to payout the repurchased shares. So you will have to add the chart for 2004-2005 by negative payouts, by negative amounts.
So let me put it strategically then. There are two ways. Either there will be a lack of liquidity or a lack of solvency. It cannot be solvency; you are at 240. So it is a matter of liquidity. I cannot imagine for it to be impossible to replace this professionally with a straight.

--------------------------------------------------------------------------------
Unidentified Audience Member    [9]
--------------------------------------------------------------------------------
Simple question, a simple answer. It is of course possible to replace it with a straight. But specifically Baloise Bank SoBa is part of our group. Baloise Bank SoBa is in an excellent position to be able to give more loans and mortgages than (technical difficulty) generating through owned resources. That is why Baloise Bank SoBa has to refinance, which it can do in the banking market or through us, the holding company, through loans.
We have been doing it for three years through loans in the holding. There loans outstanding in the amount of 600 million. September next year they will come to maturity, and they will run in the normal banking business and normal refinancing. They are being refinanced. So for the 600 million to be refinanced by September next year, and at the same time to cope with or lock in the low interest rate margin, we issued straight bond 250 million, and loans either this year or next year in the same amount will be issued. We want to have the flexibility depending on interest rate level and market conditions for it not to be a straight, but to be a very normal non-mandatory convertible or exchangeable.
This is the purpose of the conditional capital. Of course we have to show it in the annual report, but only when shares are generated. This can only be at the endpoint of a convertible. But we are clear about the uncertainty this is creating in a period of time when the market does not feel like having capital increases in the insurance market for a very transparent spread business where I strategically have to issue a straight bond.
Now to play with an equity element for this subsidiary, I think this is really a game you are playing. We are saving 2 percent on the coupon but you have a potential dilution. This game ought to be played only if, on subjective view, you are of the opinion that the share price is valued very high. Your assessment might be very relative on this point.
Well, we are dealing with a well-educated financial community here, who are in a position to assess a non-mandatory convertible and distinguish it from an equity measure. That is what it is. It is nothing else. It is not a non-mandatory convertible. Our corporate solvability (ph) is at 241 percent. So it is no problem at this point. So it is merely a matter of refinancing.

--------------------------------------------------------------------------------
Thomas Schwarzenbach,  Bank am Bellevue - Analyst    [10]
--------------------------------------------------------------------------------
Bank am Bellevue. I have got to questions on the Swiss business. There are no hard facts and figures in the outlook, but you are just giving directions. You said it is going to be better. In 2003, the loss ratio was already pretty good. So I would like to focus on the cost ratio, which was below 25 percent. Will your measures further bring down the cost ratio? And if so, are there any targets, maybe 2, 3, or 5 percentage points?

--------------------------------------------------------------------------------
Unidentified Audience Member    [11]
--------------------------------------------------------------------------------
Five percentage points? Well, perhaps; why not? 25, under 25? Martin, please.

--------------------------------------------------------------------------------
Thomas Schwarzenbach,  Bank am Bellevue - Analyst    [12]
--------------------------------------------------------------------------------
Second question; the Swiss group life business and the guarantees were 3.6 percent of technical provisions. Guaranteed surplus and credit. How will this evolve in 2004? Will it be about 40 basis points above the minimum rate? Or will there be some guarantees that will be dropped? Will the rate be lower or whatever? What will profitability be? What do you expect for it to evolve in terms of basis points, provisions?

--------------------------------------------------------------------------------
Unidentified Audience Member    [13]
--------------------------------------------------------------------------------
Thank you for the two questions. Combined ratio in the Swiss business, where is it going this year? We assume there will be an aggregate improvement of 2 to 3 percentage points on the total combined ratio. As far as the cost ratio is concerned, we do not believe that we will be clearly below 23. We are at 23.8; 23 ought to be possible. Maybe not in the first step (ph) this year, but perhaps within the next two to three years it should be possible to go down to 23. But there will be a limit at some point for the non-life business. The loss ratio we do see some potential for improvement there.
Second question regarding the 3.6; you have got to take into account here when it comes to the occupational pension scheme accounts, this is really structuring of accounts as prescribed by the Federal Office of Private Insurance two years ago. The surpluses here include the guarantees, 3.25 percent for customers, plus the risk surpluses. The surpluses generated from the risk results. That is what is meant here, and there is a bracket explaining this.
The expected value for this year on the minimum interest rate side will have 2.25. You know it's been lowered for this year, and there will be a risk surplus, perhaps, around the same value as last year. Whether it is going to be 40 basis points or around there is difficult to assess at the moment.

--------------------------------------------------------------------------------
Thomas Schwarzenbach,  Bank am Bellevue - Analyst    [14]
--------------------------------------------------------------------------------
The risk surplus is a deficit then?

--------------------------------------------------------------------------------
Unidentified Audience Member    [15]
--------------------------------------------------------------------------------
No. That is an allocation of income which was generated on risk provisions.

--------------------------------------------------------------------------------
Thomas Schwarzenbach,  Bank am Bellevue - Analyst    [16]
--------------------------------------------------------------------------------
So we can assume 2.6?

--------------------------------------------------------------------------------
Unidentified Audience Member    [17]
--------------------------------------------------------------------------------
Somewhere between 2.5 and 2.6; whether the spread is exactly 40 basis points is difficult to tell. It depends on financial markets and the risk result. But we will be somewhere in between, between 2.25 and 2.65 this year.

--------------------------------------------------------------------------------
Unidentified Audience Member    [18]
--------------------------------------------------------------------------------
(technical difficulty) investment will erode (ph) from maybe 3.87 to 3.7 or something like that. More than 1 percent will remain free, and it will be 70 basis points after-tax.

--------------------------------------------------------------------------------
Unidentified Audience Member    [19]
--------------------------------------------------------------------------------
Well, we don't know how the year will run. We have adopted premiums in the BVG business, we increased premiums beginning of the year by about 10 percent. And of this increase 5 percentage points will be allocated to cost premium, 3 percent; and 1 percent will go to conversion rates; and 1 percent will be for guaranteed minimum interest rates. But we will have to see how the year evolves.
But as a general rule, one can say that income will be far beyond a black ink naught. We expect increase in earnings in the life business. That is what we're stating. Any further questions?

--------------------------------------------------------------------------------
Stefan Schurman,  Pictet & Cie. - Analyst    [20]
--------------------------------------------------------------------------------
Pictet. I've got two questions regarding investments. You have said you were holding a share of stock of 6.2 percent. Can we expect that this will increase in 2004 or later?
Second question on fixed income. It is about 10.4 billion. That is classified as held to maturity. Is this the size that you are at ease with, or will you raise that? Specifically on the German market, I have seen that you have a relatively high allocation to surplus there; I think 350 million in the life business. On my estimate that is 3.5 percent of the technical reserve, which I consider to be rather high. Is there a specific reason for this?

--------------------------------------------------------------------------------
Martin Strobel,  Baloise Holding - CEO    [21]
--------------------------------------------------------------------------------
I will take your first two questions on investments. 6.2 percent is at the end of the year. The share ratio without holdings, that is correct. In the meantime, it's been increased by 1 percent, another percent. We believe that small increases can be possible, especially not for property but for life businesses.
Fixed income held to maturity? We've got 40 percent of the fixed-income investment of the group is nonclassified, and 44 of the insurance segment. And I am making the distinction deliberately. I think this is an appropriate value. You cannot take it much higher, so as not to restrict your leeway too much. As for Germany, Wolf?

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [22]
--------------------------------------------------------------------------------
I'm still calculating it for Germany. From page 76 of the business report, that is segment reports by regions, Germany 2003; that is line surplus and the bonuses of policyholders in Germany. The portfolio earnings are above the guarantees. The guarantees in Germany by tradition have been about 3.5 percent. In the short term it was raised to 4 percent, and then it was reduced to 3.75 percent. We now still have 2.75.
The portfolio, the investment portfolios of our three German insurance companies generate much higher earnings; 4 even to 5 percent of earnings. At least 90 percent of this is passed on to our customers. That is the case. And there is another effect which is not represented in this line, and that is the actual de facto allocation to the customers, which gives a special effect in Germany. If you keep RFBs for too long, more than three years, you have to declare them to tax.
Securitas is a well-endowed company. Then, within three years it has to report this if it is kept as an RFB for more than three years. That is one of the reasons why we can pay out such good bonuses.

--------------------------------------------------------------------------------
Unidentified Audience Member    [23]
--------------------------------------------------------------------------------
Mr. Maier (ph), Swiss First. I have a few questions to the embedded value. But, first of all I would like to thank for this increased transparency, which certainly meets the needs of the market.
First of all, the solvency cost have dropped. Why is this so? What is the interest rate that you have been counting with on your equity regarding the or compared with solvency cost? I was a bit surprised because I had expected that the embedded value would drop because you had a more conservative investment policy over the year. So, what was the impact of this on your expectations or estimates? In my opinion the effect should really have been much higher.
And then what is not contained in this embedded value is the cost savings. I am sure that the Baloise could further cut costs in the life business. You mentioned that in the occupational pension scheme business you are going to use a new IT instrument. Could you tell us what potential there is there for cost savings thanks to this new instrument?
And a last question in this connection. What is the margin on new business, 5 percent? Do you have any target for this margin on new business?

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [24]
--------------------------------------------------------------------------------
5 questions, I will try and answer them all. First one, why have the solvency costs dropped? That is a very simple answer to that. These costs, the solvency costs, arise from the lock-in effect. The effect of this is the company's capital, working capital, it is then locked in at an interest of 7.5 percent in Swiss and in 8.5 percent in Euros.
And what is the interest rate, you asked us for, on our equity? Well, 7.5 percent in Swiss francs and 8.5 percent in Europe. Because it is locked in you assume a run off for the company; and you cannot just simply take this capital out on such short call. It remains locked in. And if you have a legal quote of 3 point 303 million, which is part of the equity, then this reduces the equity by that much.
In comparison to that, you have a reduction of the locked-in equity or capital; or rather the equity drops by about 460 million. That is an order of magnitude, not a precise figure. And therefore the solvency cost moves correspondingly. I don't want to be held by the 64. This figure of 64. But our actuarial expert is nodding. When assets drop the solvency costs drop as well.
And then the third question, why does the embedded value arise if the risk in capital investment has fallen? But here we have a combination of effects. How do we calculate embedded value? Well, basic interest rate is calculated on the basis of forward rates; then it is adjusted to our portfolio in capital investment. And that gives us over time an expected return on investment, and that is how we calculate our commitments and bonuses.
But, you are right, of course. We take as an underlying basis the portfolio. Our investment portfolio. And this is reflected embedded value, which has fallen by 179 million Swiss francs. That is exactly that effect.
However, it increases over-proportionally if we can compensate for this effect, if we can reduce the surpluses which we have already done. But if we also take distinct measures. For example in the Swiss business, this is something we have done and I enumerated them, and they generate a distinct increase in the embedded value. These are operational improvement measures.
And now to question four. It was perhaps more of a comment. Cost-cutting in future, is it possible? The answer is yes. This has not been contained yet in the calculation of the embedded value. Oh, thank you for the water. So embedded value can be further enhanced. We never include the improved operating processes in the embedded value because we close the books and can start with a new slate.
The last question was margin on new inflowing capital, on new money. I don't know exactly what you meant by this. But are you talking about the ratio between embedded value and new business, divided by the annual premium equivalent? Those are the famous 15.5 divided by the 305 new premium equivalents, which gives you 5 percent.
I don't like this expression at all, because it would indicate that it is an easily measurable profitability yardstick, and it is not so, because it is very arbitrary, this value which one tries to use to calculate or evaluate the new business. So we don't really like this figure to measure profitability, and that is why we don't use it as a unit.
We would use the internal rate of return as a measuring unit. The business that is closed has to have its interest rate that is larger than the RDD by plus 1 percent; in Swiss it is 7.5 percent; so (inaudible) would have to be 8.5 percent of yield in Swiss francs. And for the euro it will be 8.5 plus 1; that is 9.5 percent of yield or profitability. And the new business has had this internal rate of return that is above this benchmark.
In the new business, 2003 we had an internal rate of an RDD plus 1.6 percent. Does that answer your questions? The five questions, rather? Very well. One more question from the participants here, than we will go to the conference calls.

--------------------------------------------------------------------------------
Unidentified Audience Member    [25]
--------------------------------------------------------------------------------
Mr. Shaup (ph) from Nauetrische (ph) Bank. I would like to come back to the embedded value. I only have two questions, if I am permitted to put two questions. The minus 303 millions from the legal quote; can you tell me how much comes on the life value in force, and how much comes from your assets?
Actually, and secondly, is the reduction of the minimum interest contained in the surplus? But have you already included this calculation 2004 in the 100 basis points? And if not, what would it amount to?

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [26]
--------------------------------------------------------------------------------
I have two possibilities. Either I start delving in my documents and I have to go into my document case, in fact 60 million are contained in the actual profit and that is the present value of future profit. That would remain untouched by the legal quote.
But the major effect comes from our assets, because the unrealized profits have to be shared out and divided in the proportion of 90 to 10; and that leads to the 460 million question; is that the correct figure?
Then the cost of solvency is then 250, I think. So you have 460 million and 250 cost of solvency. NO, I'm sorry; 223 not 250. (indiscernible) from the present value.

--------------------------------------------------------------------------------
Unidentified Audience Member    [27]
--------------------------------------------------------------------------------
The competitors show different figures, and the introduction of legal quote has led to uncertainty about how much the value in force could be. Most insurers said that they already pay out 95 percent of their surpluses and there is still a reduction. So, I was trying to understand this for the whole of Switzerland. That was the purpose of my question.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [28]
--------------------------------------------------------------------------------
The present value of future property is exactly what you said. However, the effect then has its greatest impact on the asset because 95 percent of the unrealized profit is already accounted for or satisfied.

--------------------------------------------------------------------------------
Unidentified Audience Member    [29]
--------------------------------------------------------------------------------
What about the minimum rate, the other question I put. Is it already included?

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [30]
--------------------------------------------------------------------------------
Yes. Already included is the answer. That is the reduction by the legal quote of the professional pension scheme. Thank you very much. We will now move onto questions by conference call. Could we have the first question? I'm ready for the first question.

--------------------------------------------------------------------------------
Operator    [31]
--------------------------------------------------------------------------------
Duncan Russell, (ph) SBK.

--------------------------------------------------------------------------------
Duncan Russell,  SBK - Analyst    [32]
--------------------------------------------------------------------------------
Good afternoon, and I've got a couple of questions. The first one relates to the capital increase.

--------------------------------------------------------------------------------
Unidentified Audience Member    [33]
--------------------------------------------------------------------------------
Can we turn up the volume here in the room, please?

--------------------------------------------------------------------------------
Duncan Russell,  SBK - Analyst    [34]
--------------------------------------------------------------------------------
Can you hear me now? The first one relates to the capital increase. Have you released a press release, or is it just an authorization you are seeking? So have you actually confirmed details of the capital increase, or is it just an authorization for the possibility of raising capital at some future point? And have you disclosed what type of capital you're going to raise; equity, debt and so forth?
My second question then is on the embedded value. There is a movement in the embedded value since 73-3-6 million is relating to a reduction in surplus amounts. In your speech you said this was to do with lower policyholder bonus assumption. Can you just detail what the assumption was previously and what it was afterwards? And what business area this relates to?
The next question then is also on the legal quotes. In Mr. Drunk's speech he said it was related to December's proposal. I just wanted to confirm that that was relating to the press release a couple of weeks ago by the BVG commission on the introduction of 90-10 on the investment return, plus 10 percent on the mortality and expense premium. Just to confirm that was the proposals you are talking about.
Finally does your EV your include an adjustment for the introduction or the lowering of the non-mandatory conversion rate which you say you're going to do?

--------------------------------------------------------------------------------
Unidentified Audience Member    [35]
--------------------------------------------------------------------------------
I am not sure I got all the questions. If I don't, we will come back to that. Again, can we turn up the volume of the telephone conference here in the room? It is barely hearable.
But, anyway let me start with the first point, the what you call capital increase. I would like to emphasize there is no capital increase. Baloise has a solvency of 241 percent, and we do not need any equity, and we do not intend to raise capital.
What we're doing here is we are asking the General Assembly to -- for 10 percent of conditional capital, for the purpose of being able to refinance our bonds; and one of them, a 600 million bond is maturing next year, in September next year. So that we will be able, if appropriate according to market conditions, to refinance with a non-mandatory convertible and therefore participate in low interest rates, and if it comes up good markets for just normal traditional convertible. So, that is question number one.
I would like to jump to question number three first, the legal quote and the proposal of December. It is expected that the proposal of December, that the BBFO (ph) raised, and we did -- under those assumptions we did the embedded value calculation. Now, it is in the nature of an embedded value calculation that it takes quite some time to actually do it. Just to give you an idea, Christian (ph), it takes about three to four weeks just to do an embedded value calculation.
Now for that obvious reason, it is absolutely impossible to update an embedded value calculation based upon a decision that has been taken two weeks ago. In particular, if the decision as it has been taken two weeks ago is so uncomprehensible and difficult to judge what it actually means on business going forward, as the one we are faced with. Now that is question number three.
Question number two was about the 336 and the policyholder bonus assumptions. Martin or Christian, any of you? Yes, but the question -- What is the reduction in policyholder bonuses? The question was how much exactly and what were the assumptions being in there. Can we have a microphone for our chief actuary, Christian Hippemeyer (ph).

--------------------------------------------------------------------------------
Christian Hippemeyer,  Baloise Holding - Chief Actuary    [36]
--------------------------------------------------------------------------------
We have reduced surplus to allow all the shareholders, all policies with technical interest above more than 3 percent, we have a consult (ph) surplus. I know a long-term dividend we have reduced about 25 percent.

--------------------------------------------------------------------------------
Unidentified Audience Member    [37]
--------------------------------------------------------------------------------
Question number four, I could not hear you. Could you repeat that one?

--------------------------------------------------------------------------------
Duncan Russell,  SBK - Analyst    [38]
--------------------------------------------------------------------------------
Question number four relates to the conversion rate, which you said in your speech you're going to reduce the conversion rate on the non-mandatory part of group life. Has that effect been taken into the embedded value?

--------------------------------------------------------------------------------
Christian Hippemeyer,  Baloise Holding - Chief Actuary    [39]
--------------------------------------------------------------------------------
The question was whether 5.4 percent for women and 5.8 percent for age conversion rate in the non-mandatory part has already been; the answer is yes, it has already been included.

--------------------------------------------------------------------------------
Unidentified Audience Member    [40]
--------------------------------------------------------------------------------
Second question from the telephone conference?

--------------------------------------------------------------------------------
Operator    [41]
--------------------------------------------------------------------------------
Laurent Rousseau, Credit Suisse First Boston.

--------------------------------------------------------------------------------
Laurent Rousseau,  Credit Suisse First Boston - Analyst    [42]
--------------------------------------------------------------------------------
I have a couple of questions. The first one is I really don't understand (inaudible) answer to the previous question. You don't intend to do a capital increase; but why do you do this -- why would you allow the company to have this convertible issuance? And if it is really to benefit from the low-interest level, why don't you do just trade debt? So I'm a bit confused here by your previous answer.
I have two questions apart from this one. The first one is, why do you have some DAC amortization when the profitability of life improves? That is the first question.
Second question is, what are the regulatory risks in Switzerland? That is in terms of solvency requirements that could be increased; that is in terms of legal quotes; and in individual life do you see any potentially new legal quote in individual life?
And third point is the increase of minimum guarantees from 2.25 to 3 percent. And is there sort of a big concern on regulation in Switzerland overall? Thank you.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [43]
--------------------------------------------------------------------------------
Me personally, I got question number one and three. I will check with Martin Strobel afterwards whether he got question number two. If not, you got to repeat that. So let's start with the capital increase which is not a capital increase.
I am very aware -- we are very well aware that the capital markets, the financial analysts, the investment community is very nervous and very sensitive to capital increases. This has been created by companies having made capital increases, and especially by companies having used authorized or conditional? Conditional. Having used conditional capital to actually make a capital increase.
Now, Baloise do not intend to do that. We just don't see why nervousness in the capital market should prevent us from pursuing a reasonable cost-efficient refinancing strategy. And that is why we are here, to fully commit and make you understand that this is the flexibility to be able to refinance the straight bond, either by a straight bond and not use the conditional capital; or refinance the straight bond by a simple pure convertible with a normal markup.
And it is a pure refinancing measure. I hope that I have clarified without any uncertainty in the market about capital increase. I really would appreciate not to hear the word capital increase associated with our 10 percent conditional capital (technical difficulty) interest capital.
I just want to remind that it is actually normal operating practice for basically every company to have 10 percent conditional capital. It has always been and it should always be especially for purposes like that. Baloise also has in the past a 10 percent conditional capital. So it is basically no big deal if you look at the SMI companies. Most of them actually have that.
Now, question number two, regulatory. What do we expect regulatory, and that is primary Swiss business, so I would like to refer to Martin Strobel.

--------------------------------------------------------------------------------
Martin Strobel,  Baloise Holding - CEO    [44]
--------------------------------------------------------------------------------
You posed three questions. Is there an indication that there will be rising of the minimal interest rate in the group life businesses? And then second question was, what effect do you expect from Solvency II in Switzerland? And the third question was, is there any indication that a legal quote for single life will be applied in the future?
To all three questions, right at the moment there is no indication what the actual decision will be. I start with the minimum guaranteed rate in the group collective life. In September this year, the political body will decide on this rate. One member of this particular body has indicated that he would like to raise the interest rate; but is just one member, and it is absolutely not clear in which direction the decision will go in September.
Second question about Solvency II, what capital requirements or other effects will on the life business? The regulations of Solvency II are not clear yet. There is a group of specialists now writing down the details of Solvency II. From our nowadays states (ph) I guess there is no indication possible what effects, what positive or negative effects that will have.
Legal quote single life. There is no indication yet that the legal quote single life will be applied to the single life business in Switzerland for the near future. That might happen or might not. There is no indication given so far.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [45]
--------------------------------------------------------------------------------
The two, sorry; you've got to repeat. We could not hear you.

--------------------------------------------------------------------------------
Laurent Rousseau,  Credit Suisse First Boston - Analyst    [46]
--------------------------------------------------------------------------------
The last question is why do you have some DAC amortization when the profitability of life improves?

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [47]
--------------------------------------------------------------------------------
DAC amortization, now I got that. The answer is just the opposite. I had a very long discussion with one of our actuaries. DAC works this way. The more profitability improves, the more future profitability, the higher future profitability is expected to be, and the less DAC you actually appreciate. That is the way a DAC calculation works.
So when profitability improves, you depreciate actually less DAC; whereas when profitability goes down, you actually depreciate more DAC because profitability is expected to be down in the future. It sounds at little contrary to intuition. Because you actually write DAC down in a situation where you really cannot afford; and in a situation where you can afford you don't write it down. But this is the Generally Accepted Accounting Principle for DAC, and it makes a certain sense, of course, because future profitability is going to be improved.
Now, thank you very much. I would like to take one more question out of the English conference; one more question out of the German telephone conference if we have one; and I would like after that to close the discussion with the final question of Peter Casanova here in the room. So, English one?

--------------------------------------------------------------------------------
Operator    [48]
--------------------------------------------------------------------------------
Gentlemen, there are no more questions registered from the two telephone conferences. Neither deutsche nor English.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [49]
--------------------------------------------------------------------------------
Well, Mr. Casanova, then over to you. You will bring this to the golden close.

--------------------------------------------------------------------------------
Peter Casanova,  Deutsche Bank - Analyst    [50]
--------------------------------------------------------------------------------
A follow-up on embedded value. In this 1.9 billion roughly you show in inter (ph) accounts, what is included in IRS equity on the 3.3 billion, just to work it out very smoothly?
Then to ask a second question or basically two related questions; one is combined ratio. What happened in Switzerland? We had not a very pleasing picture. Media stage (ph) I think (indiscernible) of 4 points; and now for the full year we have an improvement of almost 2 points. Did you change some accruals in the meantime, or did the business that much improve in the second half?
The third question, can you relate (ph) the combined ratio over the group? Why is your reinsurance cover so expensive, 5.7 percentage points? And what are you going to reduce that going forward?

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [51]
--------------------------------------------------------------------------------
Okay. First question. (inaudible) questions, so I have got to look at the papers here. First question was -- I won't speak Swiss-German but I need not speak English either. The conversion from IAS and embedded value, that is always very difficult. Because one thing is an accounting view and the other thing is an actuarial view of the future.
Basically, the situation is as follows. You take the IAS equity or capital and reserves and then have to deduct DAC, because DAC includes more than capital, and (technical difficulty) is automatically processed in the embedded value calculation. Then, non-realized gains on bonds have to be deducted; that is the second big item.
Next, what else do we have? Provisions for future surplus holdings has to be added. These are the large chunks. We've got consolidation. What else? And the influence of legal quote, the 300 million. That is the big drivers there.
But, you'll never be able to do one-to-one conversion because embedded value calculation is not an accounting view. The two schemes are not based on the same principles. So much for your first question.
Then, regarding your third question, I can take it briefly. That was reinsurance. The answer there is, in 2003 there were two effects in reinsurance. First of all reinsurance premiums rose markedly as opposed to the previous year, because the reinsurance ran up big losses. That was one effect, high premiums.
The second effect in 2003 was absolute absence of major claims. Absolute absence. So we have exclusively non-proportional business; and the increased premiums in 2003 are a profit to the reinsurer to the full. So the spread between gross and net is very large. As large as it can be. It's not possible for it to be beyond 100 percent.
So does it main reinsurance is too expensive? No, it doesn't need that necessarily. It only means that in 2003, reinsurance here did not help because we did not have major claims. I hope that you are not viewing your car insurance as too expensive simply because you didn't have an accident this year.
Question number two, I have to pass to Martin.

--------------------------------------------------------------------------------
Martin Strobel,  Baloise Holding - CEO    [52]
--------------------------------------------------------------------------------
Well, for the September last year, our combined ratio was 100 in the Swiss business; now it's 96. And we even announced an improvement of 2 to 3 percentage points. Are you playing any tricks, the question is?
Well, this is truly operational improvements and we can see the road to the further improvement to 2 to 3 percent.

--------------------------------------------------------------------------------
Peter Casanova,  Deutsche Bank - Analyst    [53]
--------------------------------------------------------------------------------
Am I not right in assuming that in the second semester there was a strong improvement?

--------------------------------------------------------------------------------
Martin Strobel,  Baloise Holding - CEO    [54]
--------------------------------------------------------------------------------
Yes, correct. Absolutely correct.

--------------------------------------------------------------------------------
Peter Casanova,  Deutsche Bank - Analyst    [55]
--------------------------------------------------------------------------------
Let's come back then to the reinsurance cover. How about 2002, where aggregate cost was 5.6 percent. What about that situation? Same effect?
And follow-up question would be how will you go on, moving forward with reinsurance? Will we have to expect 5 to 6 percent cost if we have no major claims?

--------------------------------------------------------------------------------
Martin Strobel,  Baloise Holding - CEO    [56]
--------------------------------------------------------------------------------
Well, one thing is relatively logical. The basis for premium calculation is earning cost. Earning cost means that claims over the past 10 years are taken. The average of the claims of the past tin years. And this is the basis for the expected claims of the next year.
If you have a claim-free such as in 2003, it really means that premiums will have to lower by 10 percent, very basically. That is inherent. Then of course, there's some market forces that would come in there. So we will press very strongly for our reinsurance premiums to be lowered accordingly.
For 2002 I would not know off my heart, but we analyzed it over a period of 10 or 11 years. Reinsurance costs here in Switzerland were 0.8 percent over a period of 11 years. Reinsurance has to be viewed over such long periods of time. Contract reinsurance at least 10 years. And then you've got facultative reinsurance, where a contract that exceeds 100 million is -- in short, against a one-off claim in 100 years, 200 to 300 million. And there you would have to apply a 100-year period to analyze.

--------------------------------------------------------------------------------
Peter Casanova,  Deutsche Bank - Analyst    [57]
--------------------------------------------------------------------------------
Well, it is asking too much to view the stock exchange over a period of 100 years.

--------------------------------------------------------------------------------
Wolfgang Drunk,  Baloise Holding - CFO    [58]
--------------------------------------------------------------------------------
Yes, but that is not really what we're talking about, is it? On this I would like to thank you; thank all those involved. And as soon as the microphones for the Internet will have been closed, I will welcome everybody in here and ask everybody in here to join us for drinks. Thank you.

--------------------------------------------------------------------------------
Operator    [59]
--------------------------------------------------------------------------------
Ladies and gentlemen, the conference call is now over. You may now disconnect your telephones. Thank you for joining. Goodbye.








--------------------------------------------------------------------------------
Definitions
--------------------------------------------------------------------------------
PRELIMINARY TRANSCRIPT: "Preliminary Transcript" indicates that the
Transcript has been published in near real-time by an experienced
professional transcriber.  While the Preliminary Transcript is highly
accurate, it has not been edited to ensure the entire transcription
represents a verbatim report of the call.

EDITED TRANSCRIPT: "Edited Transcript" indicates that a team of professional
editors have listened to the event a second time to confirm that the
content of the call has been transcribed accurately and in full.

--------------------------------------------------------------------------------
Disclaimer
--------------------------------------------------------------------------------
Thomson Reuters reserves the right to make changes to documents, content, or other
information on this web site without obligation to notify any person of
such changes.

In the conference calls upon which Event Transcripts are based, companies
may make projections or other forward-looking statements regarding a variety
of items. Such forward-looking statements are based upon current
expectations and involve risks and uncertainties. Actual results may differ
materially from those stated in any forward-looking statement based on a
number of important factors and risks, which are more specifically
identified in the companies' most recent SEC filings. Although the companies
may indicate and believe that the assumptions underlying the forward-looking
statements are reasonable, any of the assumptions could prove inaccurate or
incorrect and, therefore, there can be no assurance that the results
contemplated in the forward-looking statements will be realized.

THE INFORMATION CONTAINED IN EVENT TRANSCRIPTS IS A TEXTUAL REPRESENTATION
OF THE APPLICABLE COMPANY'S CONFERENCE CALL AND WHILE EFFORTS ARE MADE TO
PROVIDE AN ACCURATE TRANSCRIPTION, THERE MAY BE MATERIAL ERRORS, OMISSIONS,
OR INACCURACIES IN THE REPORTING OF THE SUBSTANCE OF THE CONFERENCE CALLS.
IN NO WAY DOES THOMSON REUTERS OR THE APPLICABLE COMPANY ASSUME ANY RESPONSIBILITY FOR ANY INVESTMENT OR OTHER
DECISIONS MADE BASED UPON THE INFORMATION PROVIDED ON THIS WEB SITE OR IN
ANY EVENT TRANSCRIPT. USERS ARE ADVISED TO REVIEW THE APPLICABLE COMPANY'S
CONFERENCE CALL ITSELF AND THE APPLICABLE COMPANY'S SEC FILINGS BEFORE
MAKING ANY INVESTMENT OR OTHER DECISIONS.
--------------------------------------------------------------------------------
Copyright 2019 Thomson Reuters. All Rights Reserved.
--------------------------------------------------------------------------------
"""


# def test_extract_gpt_training_dataset_from_preprocessed_transcripts(test_transcript):
#     country_names_set = set(["swiss", "belgium"])

#     transcript_row = pd.DataFrame(
#         {
#             "Transcript_ID": 1,
#             "Preprocessed_Transcript_Step_1": preprocess_transcript_text(
#                 test_transcript
#             ),
#         },
#         index=[0],
#     )

#     actual_result = extract_gpt_training_dataset_from_preprocessed_transcripts(
#         transcript_row, country_names_set
#     )

#     assert len(actual_result) == 91


# def test_get_index_where_words_occur():
#     set_of_words = {"hello", "world"}
#     text = "hello world, hello again"
#     expected_output = [0, 6, 13]
#     assert get_index_where_words_occur(set_of_words, text) == expected_output
