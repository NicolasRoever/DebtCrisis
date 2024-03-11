import pandas as pd

import re

from src.debt_crisis.sentiment_index.clean_sentiment_data import extract_date_from_transcript


test_transcript = r"""
Thomson Reuters StreetEvents Event Transcript
E D I T E D   V E R S I O N

Hanover Capital Mortgage Holdings to Hold Conference Call to Update 2004 Fourth-Quarter and Year-End Financial Results
APRIL 01, 2005 / 4:00PM GMT

================================================================================
Corporate Participants
================================================================================

 * John Burchett
   Hanover Capital Mortgage Holdings - CEO
 * Irma Tavares
   Hanover Capital Mortgage Holdings - COO
 * Holly Loux
   Hanover Capital Mortgage Holdings - CFO

================================================================================
Conference Call Participiants
================================================================================

 * Joe Stieven
   Stifel Nicolaus and Company - Analyst

================================================================================
Presentation
--------------------------------------------------------------------------------
Operator    [1]
--------------------------------------------------------------------------------
Greetings ladies and gentlemen, and welcome to the Hanover Capital Mortgage fourth quarter earnings update conference call.  (OPERATOR INSTRUCTIONS)  It is now my pleasure to introduce your host, Mr. John Burchett, chief executive officer of Hanover Capital Mortgage.  Thank you, Mr. Burchett, you may begin.

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [2]
--------------------------------------------------------------------------------
Good morning and thank you.  Thank you all for joining the call this morning.  I am joined here with Irma Tavares, our chief operating officer, and Holly Loux, our chief financial officer, for this call, and we just wanted to have, again, a brief call.  We had one prior to this to announce the top line earnings and we did our filing last night, so we wanted to have a more regular call where we have a press release out with a balance sheet income statement for people to have as part of the call.
Just to review the earnings, the fourth quarter was a good, solid quarter for us.  We were 33 cents in earnings.  We paid a dividend of 30 cents based on stock price recently.  Our annual dividend rate of $1.20 equates to about 11.3 percent yield on that stock price, which we consider to be a good yield even in today's market, where rates have gone up somewhat.  It's a nice spread over the 5- or 10-year treasury rate.
Our basic business remains good.  Our business, as most of you know, is to invest in the credit side of the prime residential mortgage market, and we have had very little, almost none, in terms of losses over the past few years.  I think last year we had on those subordinate pieces less than $1,000 in losses, and those really aren't credit losses, per se, but tend to be losses based on prepayments of loans, that there is some interest that comes through to the subordinate piece that doesn't get paid, so it was really a negligible amount of loss.  So the credit remains good.  Our product remains on the prime credit side with low LTVs and high FICO, or credit scores on those loans, and even with all the talk and buzz about housing bubbles, we still think we're well protected in terms of our credit position in those loans, and we continue to push on that strategy, is what we want to do.  So we think there are good returns and good credit risk still available in that market.
We did file our Form 10-K with the SEC last night.  It was delayed, but it was in the grace period that's allowed in those filings, and as we had announced, there are a couple of reasons that it took us a little longer this year.  Clearly, the Sarbanes-Oxley reporting is a burden for all of corporate America that reports publicly.  I think it's particularly burdensome on smaller companies both in terms of time and effort of the principals involved and in the actual cost of getting it done.  But we did get it done.  We filed with a clean SOX opinion -- not to be a pun on clean SOX, but it was an opinion with no negatives in it.  And, again, we went through a lot of work to get that done, a lot of expense to get it done, but we did get it done, and now we have the ground work and the basic policies documented.  It's not so much that we needed new policies, but we needed documentation and testing of those policies to meet the requirements of Sarbanes-Oxley.  And now those are in place and we expect that the cost to us of continual compliance will decrease over this year and the future years, although there still is a requirement to do continual testing and updating of documentation.
It also points out our need to continue to grow the capital base of the Company, and obviously we need to be a larger company.  And in that vein, we did announce previously that we issued $20 million worth of preferred stock, one route to raising capital, and we will continue to explore that and other routes as we go through this year as one of our primary goals.  So our goals generally would be to remain investing in the prime credit markets of the residential mortgage side and to grow our capital base over the year.
Basically, that's all I had to go over.  We'll open it up for questions now and be happy to answer any questions anybody has.

--------------------------------------------------------------------------------
Operator    [3]
--------------------------------------------------------------------------------
(OPERATOR INSTRUCTIONS)  Our first question is coming from Joe Stieven of Stifel Nicolaus.  Please proceed with your question.

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [4]
--------------------------------------------------------------------------------
Hi, John, good morning.

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [5]
--------------------------------------------------------------------------------
Good morning, Joe, how are you?

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [6]
--------------------------------------------------------------------------------
Very well.  John, you guys did announce that you raised $20 million, I guess in the trust preferred market.

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [7]
--------------------------------------------------------------------------------
	Yes.

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [8]
--------------------------------------------------------------------------------
	When you look at that money, will you leverage that just like common equity?  That's question number one.  Question number two is, what do you think of the ratio of quote, unquote, preferred -- let's just call that preferred.  What type of ratio of preferred to common do you think is appropriate for you guys?  Just sort of on a go-forward basis.

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [9]
--------------------------------------------------------------------------------
Yes.  On the first question, we look at this as a 30-year deal, and particularly at my age we look at this as permanent capital.  So we would intend to leverage it at least in the early years of that as we do equity capital.  So we would put it to work similar to how we do equity capital.  The second question is what are the ratios.  I think it's a little bit dictated by the market, but I would guess in general we could probably do another issue of this or similar size and that would probably be about the limit at our current common equity base, and that would probably put us in the 40 percent, maybe, but permanent capital.  And in this case, we fixed the rate on this for five years, so I think that's a reasonable level.  So I think we could probably do one more of this size or maybe a little bigger, $20 to $25 million in preferred before we would need to add common to support more on the preferred side.

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [10]
--------------------------------------------------------------------------------
John, also, give us the exact -- can you give us some better details.  You said you fixed it for five years; what was the rate you fixed it at?

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [11]
--------------------------------------------------------------------------------
It was just a little over 8.50.  The basic spread was -- I'm working off the top of my head now, but I think it's 4.25 LIBOR, and we swapped it, we basically swapped inside the deal.  The swap is not on our books, but it's inside the transaction was swapped floating for fixed, and at the time about the same spread over the five years, so it ended up, I think it's at 8.55 actual.

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [12]
--------------------------------------------------------------------------------
And so after five years, then, you'll be at floating at one month LIBOR, or -- is that what it will be?

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [13]
--------------------------------------------------------------------------------
After five years, it floats to one month LIBOR -- actually, they're telling me the actual rate was 8.51, not 55.  Yes, we float to one month LIBOR -- actually, I take that back -- it's three month LIBOR, and we floated 4.25 over that.  And one of the reasons we picked five years is, we can prepay it without penalty at that point in time, not that we would have an intention to doing that, but if rates were really strange and it didn't make sense to have this piece in their capital structure, we do have the ability to repay it at the five-year date without any prepayment penalty.

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [14]
--------------------------------------------------------------------------------
And, John, in today's current market, you know, because you used to always talk about the ROEs you could from the investment market.  If you have to go in theory employ this capital right now, can you give us some thoughts on what you think the appropriate returns are that you can see in the market?

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [15]
--------------------------------------------------------------------------------
I think on a leverage basis in today's market, we're still in the mid-teens of leveraged capital.

--------------------------------------------------------------------------------
Joe Stieven,  Stifel Nicolaus and Company - Analyst    [16]
--------------------------------------------------------------------------------
Okay.  Okay.  Thank you, John.

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [17]
--------------------------------------------------------------------------------
Okay.  Thanks, Joe.

--------------------------------------------------------------------------------
Operator    [18]
--------------------------------------------------------------------------------
(OPERATOR INSTRUCTIONS)  We show no further questions in the queue at this time.  I'd like to turn the floor back over to our speakers.

--------------------------------------------------------------------------------
John Burchett,  Hanover Capital Mortgage Holdings - CEO    [19]
--------------------------------------------------------------------------------
Okay.  Again, I thank everybody for joining us.  I look forward to talking to you at the next call.  Thanks again.  Bye.

--------------------------------------------------------------------------------
Operator    [20]
--------------------------------------------------------------------------------
This concludes today's conference.  Thank you for your participation.







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


def test_extract_year_from_transcript_expected_behaviour(test_transcript=test_transcript):

    expected_result = "APRIL 01, 2005"
    actual_result = extract_date_from_transcript(test_transcript)

    assert actual_result == expected_result

