### Web Scraping 2019 NFL game logs

In this project there is an example of web scraping using R (robotstxt
and rvest).

Loading Packages

``` r
library(robotstxt)
library(rvest)
```

    ## Loading required package: xml2

``` r
library(stringr)
library(plyr)
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.0 --

    ## v ggplot2 3.3.2     v purrr   0.3.4
    ## v tibble  3.0.4     v dplyr   1.0.2
    ## v tidyr   1.1.2     v forcats 0.5.0
    ## v readr   1.3.1

    ## Warning: package 'tibble' was built under R version 4.0.3

    ## Warning: package 'dplyr' was built under R version 4.0.3

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::arrange()        masks plyr::arrange()
    ## x purrr::compact()        masks plyr::compact()
    ## x dplyr::count()          masks plyr::count()
    ## x dplyr::failwith()       masks plyr::failwith()
    ## x dplyr::filter()         masks stats::filter()
    ## x readr::guess_encoding() masks rvest::guess_encoding()
    ## x dplyr::id()             masks plyr::id()
    ## x dplyr::lag()            masks stats::lag()
    ## x dplyr::mutate()         masks plyr::mutate()
    ## x purrr::pluck()          masks rvest::pluck()
    ## x dplyr::rename()         masks plyr::rename()
    ## x dplyr::summarise()      masks plyr::summarise()
    ## x dplyr::summarize()      masks plyr::summarize()

Confirming using robotstxt that it is acceptable to scrape from the
website.

``` r
base_url = "https://www.pro-football-reference.com"
paths_allowed(base_url)
```

    ##  www.pro-football-reference.com

    ## [1] TRUE

Writing a script that will output the links to game log data for all 30
teams. As an example, the link for the Dallas Cowboys is
<a href="https://www.pro-football-reference.com/teams/dal/2019.htm" class="uri">https://www.pro-football-reference.com/teams/dal/2019.htm</a>.

``` r
####Modify file paths to get game logs
pages <- read_html(file.path(base_url)) %>%
  html_nodes("#teams .left a") %>%
  html_attr("href") %>%
  file.path(base_url, .) %>%
  str_replace("2020", "2019") %>%
  str_replace(".htm", "/gamelog/")

pages
```

    ##  [1] "https://www.pro-football-reference.com//teams/buf/2019/gamelog/"
    ##  [2] "https://www.pro-football-reference.com//teams/mia/2019/gamelog/"
    ##  [3] "https://www.pro-football-reference.com//teams/nwe/2019/gamelog/"
    ##  [4] "https://www.pro-football-reference.com//teams/nyj/2019/gamelog/"
    ##  [5] "https://www.pro-football-reference.com//teams/pit/2019/gamelog/"
    ##  [6] "https://www.pro-football-reference.com//teams/rav/2019/gamelog/"
    ##  [7] "https://www.pro-football-reference.com//teams/cle/2019/gamelog/"
    ##  [8] "https://www.pro-football-reference.com//teams/cin/2019/gamelog/"
    ##  [9] "https://www.pro-football-reference.com//teams/oti/2019/gamelog/"
    ## [10] "https://www.pro-football-reference.com//teams/clt/2019/gamelog/"
    ## [11] "https://www.pro-football-reference.com//teams/htx/2019/gamelog/"
    ## [12] "https://www.pro-football-reference.com//teams/jax/2019/gamelog/"
    ## [13] "https://www.pro-football-reference.com//teams/kan/2019/gamelog/"
    ## [14] "https://www.pro-football-reference.com//teams/rai/2019/gamelog/"
    ## [15] "https://www.pro-football-reference.com//teams/sdg/2019/gamelog/"
    ## [16] "https://www.pro-football-reference.com//teams/den/2019/gamelog/"
    ## [17] "https://www.pro-football-reference.com//teams/was/2019/gamelog/"
    ## [18] "https://www.pro-football-reference.com//teams/nyg/2019/gamelog/"
    ## [19] "https://www.pro-football-reference.com//teams/dal/2019/gamelog/"
    ## [20] "https://www.pro-football-reference.com//teams/phi/2019/gamelog/"
    ## [21] "https://www.pro-football-reference.com//teams/gnb/2019/gamelog/"
    ## [22] "https://www.pro-football-reference.com//teams/chi/2019/gamelog/"
    ## [23] "https://www.pro-football-reference.com//teams/min/2019/gamelog/"
    ## [24] "https://www.pro-football-reference.com//teams/det/2019/gamelog/"
    ## [25] "https://www.pro-football-reference.com//teams/nor/2019/gamelog/"
    ## [26] "https://www.pro-football-reference.com//teams/tam/2019/gamelog/"
    ## [27] "https://www.pro-football-reference.com//teams/car/2019/gamelog/"
    ## [28] "https://www.pro-football-reference.com//teams/atl/2019/gamelog/"
    ## [29] "https://www.pro-football-reference.com//teams/sea/2019/gamelog/"
    ## [30] "https://www.pro-football-reference.com//teams/ram/2019/gamelog/"
    ## [31] "https://www.pro-football-reference.com//teams/crd/2019/gamelog/"
    ## [32] "https://www.pro-football-reference.com//teams/sfo/2019/gamelog/"

Writing a script that will grab the game log data for a specific team
and its respective opponent.

``` r
gamelog.temp <- read_html(pages[1]) %>%
  html_nodes("#gamelog2019") %>%
  html_table(header=T)
gamelog.temp <- gamelog.temp[[1]]
opp.gamelog.temp <- read_html(pages[1]) %>%
  html_nodes("#gamelog_opp2019") %>%
  html_table(header=T)
opp.gamelog.temp <- opp.gamelog.temp[[1]]
```

Cleaning up the data - remove redundant rows, ensure column names don’t
include special characters, ensure all numeric columns are actually
numeric.

``` r
colnames(gamelog.temp) <- c("week", "day", "date", "boxscore_word", "game_outcome", "overtime", "game_location", "opp", "points_scored", "points_allowed", "pass.cmp", "pass.att", "pass.yds", "pass.TD", "pass.int", "pass.sk", "pass.yds", "pass.Y/A", "pass.NY/A", "pass.cmp%", "pass.rate", "rush.att", "rush.yds", "rush.Y/A", "rush.TD", "FGM", "FGA","XPM", "XPA", "punt.pnt", "punt.yds", "3Dconv", "3DAtt", "4DConv", "4DAtt", "ToP")
colnames(opp.gamelog.temp) <- c("week", "day", "date", "boxscore_word", "game_outcome", "overtime", "game_location", "opp", "points_scored", "points_allowed", "opp.pass.cmp", "opp.pass.att", "opp.pass.yds", "opp.pass.TD", "opp.pass.int", "opp.pass.sk", "opp.pass.yds", "opp.pass.Y/A", "opp.pass.NY/A", "opp.pass.cmp%", "opp.pass.rate", "opp.rush.att", "opp.rush.yds", "opp.rush.Y/A", "opp.rush.TD", "opp.FGM", "opp.FGA","opp.XPM", "opp.XPA", "opp.punt.pnt", "opp.punt.yds", "opp.3Dconv", "opp.3DAtt", "opp.4DConv", "opp.4DAtt", "opp.ToP")

gamelog.temp <- gamelog.temp[-1,]
gamelog.temp[,c(11:25)] <- sapply(gamelog.temp[,c(9:35)], as.numeric)
gamelog.temp$team <- str_sub(pages[1], 47, 49)

opp.gamelog.temp <- opp.gamelog.temp[-1,]
opp.gamelog.temp[,c(11:25)] <- sapply(opp.gamelog.temp[,c(9:35)], as.numeric)
opp.gamelog.temp$team <- str_sub(pages[1], 47, 49)
```

Concatenating Gamelog and Opposition Gamelog dataframes

``` r
nwe_gamelog <- join(gamelog.temp, opp.gamelog.temp,type="full")
head(nwe_gamelog)
```

    ##   week day         date boxscore_word game_outcome overtime game_location
    ## 1    1 Sun  September 8      boxscore            W                      @
    ## 2    2 Sun September 15      boxscore            W                      @
    ## 3    3 Sun September 22      boxscore            W                       
    ## 4    4 Sun September 29      boxscore            L                       
    ## 5    5 Sun    October 6      boxscore            W                      @
    ## 6    7 Sun   October 20      boxscore            W                       
    ##                    opp points_scored points_allowed pass.cmp pass.att pass.yds
    ## 1        New York Jets            17             16       17       16       24
    ## 2      New York Giants            28             14       28       14       19
    ## 3   Cincinnati Bengals            21             17       21       17       23
    ## 4 New England Patriots            10             16       10       16       22
    ## 5     Tennessee Titans            14              7       14        7       23
    ## 6       Miami Dolphins            31             21       31       21       16
    ##   pass.TD pass.int pass.sk pass.Y/A pass.NY/A pass.cmp% pass.rate rush.att
    ## 1      37      242       1        1        12       6.9       6.4     64.9
    ## 2      30      237       1        3        16       8.4       7.2     63.3
    ## 3      36      241       1        1         2       6.8       6.5     63.9
    ## 4      44      240       0        5        40       6.4       4.9     50.0
    ## 5      32      204       2        4        15       6.8       5.7     71.9
    ## 6      26      188       2        2        14       7.8       6.7     61.5
    ##   rush.yds rush.Y/A rush.TD FGM FGA XPM XPA punt.pnt punt.yds 3Dconv 3DAtt
    ## 1     69.9       25     128   1   1   2   2        3      130      5    10
    ## 2     98.9       34     151   0   0   4   4        7      325      5    13
    ## 3     80.9       36     175   2   3   1   1        5      173      5    13
    ## 4     28.6       22     135   1   2   1   1        6      180      2    13
    ## 5     96.4       27     109   0   0   2   2        6      300      4    13
    ## 6    109.1       23     117   3   3   2   2        3      127      3    10
    ##   4DConv 4DAtt   ToP team opp.pass.cmp opp.pass.att opp.pass.yds opp.pass.TD
    ## 1      0     1 27:59  buf           17           16           28          41
    ## 2      0     0 32:38  buf           28           14           26          45
    ## 3      0     0 36:54  buf           21           17           20          36
    ## 4      1     2 32:40  buf           10           16           18          39
    ## 5      1     2 31:12  buf           14            7           13          22
    ## 6      0     0 26:29  buf           31           21           23          35
    ##   opp.pass.int opp.pass.sk opp.pass.Y/A opp.pass.NY/A opp.pass.cmp%
    ## 1          155           1            4            20           4.3
    ## 2          241           1            1             9           5.6
    ## 3          240           1            2            10           6.9
    ## 4          150           0            0             0           3.8
    ## 5          150           0            5            33           8.3
    ## 6          272           1            1            10           8.1
    ##   opp.pass.rate opp.rush.att opp.rush.yds opp.rush.Y/A opp.rush.TD opp.FGM
    ## 1           3.4         68.3         82.9           21          68       0
    ## 2           5.2         57.8         61.4           20         129       0
    ## 3           6.3         55.6         62.3           19          66       1
    ## 4           3.8         46.2         45.9           23          74       1
    ## 5           5.6         59.1         79.7           27         102       0
    ## 6           7.6         65.7         86.8           30         109       0
    ##   opp.FGA opp.XPM opp.XPA opp.punt.pnt opp.punt.yds opp.3Dconv opp.3DAtt
    ## 1       1       0       1            7          304          7        17
    ## 2       1       2       2            5          242          3        12
    ## 3       1       2       2            6          243          3        11
    ## 4       1       1       2            9          435          5        18
    ## 5       4       1       1            6          300          4        14
    ## 6       0       3       3            4          196          7        13
    ##   opp.4DConv opp.4DAtt opp.ToP
    ## 1          1         2   32:01
    ## 2          1         2   27:22
    ## 3          0         0   23:06
    ## 4          0         0   27:20
    ## 5          0         0   28:48
    ## 6          2         2   33:31

Systematically collecting all game logs for all the teams.

``` r
####First, create a placeholder for each team's game logs
gamelog.team <- list()

####Function
for(i in 1:length(pages)){
  
  gamelog.temp <- read_html(pages[i]) %>%
    html_nodes("#gamelog2019") %>%
    html_table(header=T)
  gamelog.temp <- gamelog.temp[[1]]
  opp.gamelog.temp <- read_html(pages[i]) %>%
    html_nodes("#gamelog_opp2019") %>%
    html_table(header=T)
  opp.gamelog.temp <- opp.gamelog.temp[[1]]
  
  colnames(gamelog.temp) <- c("week", "day", "date", "boxscore_word", "game_outcome", "overtime", "game_location", "opp", "points_scored", "points_allowed", "pass.cmp", "pass.att", "pass.yds", "pass.TD", "pass.int", "pass.sk", "pass.yds", "pass.Y/A", "pass.NY/A", "pass.cmp%", "pass.rate", "rush.att", "rush.yds", "rush.Y/A", "rush.TD", "FGM", "FGA","XPM", "XPA", "punt.pnt", "punt.yds", "3Dconv", "3DAtt", "4DConv", "4DAtt", "ToP")
  colnames(opp.gamelog.temp) <- c("week", "day", "date", "boxscore_word", "game_outcome", "overtime", "game_location", "opp", "points_scored", "points_allowed", "opp.pass.cmp", "opp.pass.att", "opp.pass.yds", "opp.pass.TD", "opp.pass.int", "opp.pass.sk", "opp.pass.yds", "opp.pass.Y/A", "opp.pass.NY/A", "opp.pass.cmp%", "opp.pass.rate", "opp.rush.att", "opp.rush.yds", "opp.rush.Y/A", "opp.rush.TD", "opp.FGM", "opp.FGA","opp.XPM", "opp.XPA", "opp.punt.pnt", "opp.punt.yds", "opp.3Dconv", "opp.3DAtt", "opp.4DConv", "opp.4DAtt", "opp.ToP")
  
  gamelog.temp <- gamelog.temp[-1,]
  gamelog.temp[,c(11:25)] <- sapply(gamelog.temp[,c(9:35)], as.numeric)
  gamelog.temp$team <- str_sub(pages[i], 47, 49)
  
  opp.gamelog.temp <- opp.gamelog.temp[-1,]
  opp.gamelog.temp[,c(11:25)] <- sapply(opp.gamelog.temp[,c(9:35)], as.numeric)
  opp.gamelog.temp$team <- str_sub(pages[i], 47, 49)
  
  gamelog <- join(gamelog.temp, opp.gamelog.temp,type="full")
  gamelog.team[[i]] <- gamelog
}
```

Compilation of all game logs into a single data frame.

``` r
gamelogs.all <- rbind.fill(gamelog.team)
```

Creating the final data set “nfl2019\_gamelogs”. Output of the results
for str(nfl2019\_gamelogs)

``` r
nfl2019_gamelogs <- gamelogs.all
str(nfl2019_gamelogs)
```

    ## 'data.frame':    512 obs. of  61 variables:
    ##  $ week          : chr  "1" "2" "3" "4" ...
    ##  $ day           : chr  "Sun" "Sun" "Sun" "Sun" ...
    ##  $ date          : chr  "September 8" "September 15" "September 22" "September 29" ...
    ##  $ boxscore_word : chr  "boxscore" "boxscore" "boxscore" "boxscore" ...
    ##  $ game_outcome  : chr  "W" "W" "W" "L" ...
    ##  $ overtime      : chr  "" "" "" "" ...
    ##  $ game_location : chr  "@" "@" "" "" ...
    ##  $ opp           : chr  "New York Jets" "New York Giants" "Cincinnati Bengals" "New England Patriots" ...
    ##  $ points_scored : chr  "17" "28" "21" "10" ...
    ##  $ points_allowed: chr  "16" "14" "17" "16" ...
    ##  $ pass.cmp      : num  17 28 21 10 14 31 13 24 16 37 ...
    ##  $ pass.att      : num  16 14 17 16 7 21 31 9 19 20 ...
    ##  $ pass.yds      : num  24 19 23 22 23 16 16 14 22 21 ...
    ##  $ pass.TD       : num  37 30 36 44 32 26 34 20 41 33 ...
    ##  $ pass.int      : num  242 237 241 240 204 188 155 146 260 256 ...
    ##  $ pass.sk       : num  1 1 1 0 2 2 2 1 0 3 ...
    ##  $ pass.Y/A      : num  1 3 1 5 4 2 4 2 1 0 ...
    ##  $ pass.NY/A     : num  12 16 2 40 15 14 14 14 6 0 ...
    ##  $ pass.cmp%     : num  6.9 8.4 6.8 6.4 6.8 7.8 5 8 6.5 7.8 ...
    ##  $ pass.rate     : num  6.4 7.2 6.5 4.9 5.7 6.7 4.1 6.6 6.2 7.8 ...
    ##  $ rush.att      : num  64.9 63.3 63.9 50 71.9 61.5 47.1 70 53.7 63.6 ...
    ##  $ rush.yds      : num  69.9 98.9 80.9 28.6 96.4 ...
    ##  $ rush.Y/A      : num  25 34 36 22 27 23 20 39 20 34 ...
    ##  $ rush.TD       : num  128 151 175 135 109 117 98 122 84 168 ...
    ##  $ FGM           : chr  "1" "0" "2" "1" ...
    ##  $ FGA           : chr  "1" "0" "3" "2" ...
    ##  $ XPM           : chr  "2" "4" "1" "1" ...
    ##  $ XPA           : chr  "2" "4" "1" "1" ...
    ##  $ punt.pnt      : chr  "3" "7" "5" "6" ...
    ##  $ punt.yds      : chr  "130" "325" "173" "180" ...
    ##  $ 3Dconv        : chr  "5" "5" "5" "2" ...
    ##  $ 3DAtt         : chr  "10" "13" "13" "13" ...
    ##  $ 4DConv        : chr  "0" "0" "0" "1" ...
    ##  $ 4DAtt         : chr  "1" "0" "0" "2" ...
    ##  $ ToP           : chr  "27:59" "32:38" "36:54" "32:40" ...
    ##  $ team          : chr  "buf" "buf" "buf" "buf" ...
    ##  $ opp.pass.cmp  : num  17 28 21 10 14 31 13 24 16 37 ...
    ##  $ opp.pass.att  : num  16 14 17 16 7 21 31 9 19 20 ...
    ##  $ opp.pass.yds  : num  28 26 20 18 13 23 17 15 26 32 ...
    ##  $ opp.pass.TD   : num  41 45 36 39 22 35 24 22 38 45 ...
    ##  $ opp.pass.int  : num  155 241 240 150 150 272 153 116 221 280 ...
    ##  $ opp.pass.sk   : num  1 1 1 0 0 1 1 0 2 0 ...
    ##  $ opp.pass.Y/A  : num  4 1 2 0 5 1 3 4 2 7 ...
    ##  $ opp.pass.NY/A : num  20 9 10 0 33 10 19 28 17 43 ...
    ##  $ opp.pass.cmp% : num  4.3 5.6 6.9 3.8 8.3 8.1 7.2 6.5 6.3 7.2 ...
    ##  $ opp.pass.rate : num  3.4 5.2 6.3 3.8 5.6 7.6 5.7 4.5 5.5 5.4 ...
    ##  $ opp.rush.att  : num  68.3 57.8 55.6 46.2 59.1 65.7 70.8 68.2 68.4 71.1 ...
    ##  $ opp.rush.yds  : num  82.9 61.4 62.3 45.9 79.7 ...
    ##  $ opp.rush.Y/A  : num  21 20 19 23 27 30 41 23 26 13 ...
    ##  $ opp.rush.TD   : num  68 129 66 74 102 109 218 127 147 23 ...
    ##  $ opp.FGM       : chr  "0" "0" "1" "1" ...
    ##  $ opp.FGA       : chr  "1" "1" "1" "1" ...
    ##  $ opp.XPM       : chr  "0" "2" "2" "1" ...
    ##  $ opp.XPA       : chr  "1" "2" "2" "2" ...
    ##  $ opp.punt.pnt  : chr  "7" "5" "6" "9" ...
    ##  $ opp.punt.yds  : chr  "304" "242" "243" "435" ...
    ##  $ opp.3Dconv    : chr  "7" "3" "3" "5" ...
    ##  $ opp.3DAtt     : chr  "17" "12" "11" "18" ...
    ##  $ opp.4DConv    : chr  "1" "1" "0" "0" ...
    ##  $ opp.4DAtt     : chr  "2" "2" "0" "0" ...
    ##  $ opp.ToP       : chr  "32:01" "27:22" "23:06" "27:20" ...
