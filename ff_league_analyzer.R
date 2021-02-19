# Data Setup --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(ggrepel)
library(gridExtra)
library(lubridate)
library(formattable)
library(knitr)
library(kableExtra)
library(htmltools)
library(webshot)
library(leaps)
library(glmnet)
library(reticulate)
library(data.table)
library(reshape2)
library(scales)
library(stringr)
library(forcats)
source_python("C:/Users/hirsh/OneDrive/Desktop/Data Science Stuff/Projects/FF Leagues/FF Analysis/all_functions.py")

# set parameters
#league_id <- 1382012
league_id <- 329185
season <- 2020
swid = ''
espn = ''

d <- load_data(league_id, season, swid = "", espn = "")
params <- get_params(d)
league_size <- params[[1]]

# loop doesnt work???
# #ros_proj <- get_ros_projections(d)
# pr <- data.frame()
# for (yr in seq(2018,season)) {
#     d <- load_data(league_id, yr, swid="", espn="")
#     regular_season_end <- d$settings$scheduleSettings$matchupPeriodCount
#     df <- power_rank(d, league_id, yr)
#     df <- df[df$week <= regular_season_end,]
#     df$season <- yr
#     pr <- rbind(pr, df)
# }

# prior <- pr[pr$season < season,]
# current <- pr[pr$season == season,]
# current$team <- as.character(current$team)

# # Get simulation results
# Need to translate this from python to r
# n=1000
# week_sim = sim_week(n_sim=n)
# sim_week = week_sim[0]
# sim_week_scores = week_sim[1].reset_index(drop=True)
# 
# if matchup_week < regular_season_end:
#     season_sim = sim_season(n_sim=n)
#     sim_season_table = season_sim[0]
#     sim_season_results = season_sim[1]
# else:
#     sim_season_table = sim_season(n_sim=n)
# 
# sim_playoffs = sim_playoffs(n_sim=n)


# Power Ranking -----------------------------------------------------------
ssn <- 2018
d <- load_data(1382012, ssn, swid, espn)
regular_season_end <- d$settings$scheduleSettings$matchupPeriodCount
df <- power_rank(d, league_id, ssn)
df <- df[df$week <= regular_season_end,]
df$season <- ssn
pr <- df

ssn <- 2019
d <- load_data(1382012, ssn, swid, espn)
regular_season_end <- d$settings$scheduleSettings$matchupPeriodCount
df <- power_rank(d, league_id, ssn)
df <- df[df$week <= regular_season_end,]
df$season <- ssn
pr <- rbind(pr, df)

ssn <- 2020
d <- load_data(league_id, ssn, swid="", espn="")
regular_season_end <- d$settings$scheduleSettings$matchupPeriodCount
df <- power_rank(d, league_id, ssn)
df <- df[df$week <= regular_season_end,]
df$season <- ssn
pr <- rbind(pr, df)

prior <- pr[pr$season < 2020,]
current <- pr[pr$season == season,]
current$team <- as.character(current$team)


get_weights <- function(){
    # Get weights using previous seasons as training
    # split into training and testing
    set.seed(7515)
    train=sample(c(T,F), nrow(prior), rep=TRUE, p=c(0.8,0.2))
    test=!train
    
    X <- model.matrix(power_rank_score ~ win_index + luck_index + consistency_index + score_index + season_index, data = prior)[,-1]
    y <- prior$power_rank_score
    
    set.seed(1)
    cv <- cv.glmnet(X[train,],y[train],alpha=0.5)
    
    # Find the value of lambda from cross-validation 
    # and see how the resulting model works on the test data
    bestlambda <- cv$lambda.min
    elastic.mod.betas <- coef(cv, s=bestlambda)
    pred.elastic <- predict(cv, s=bestlambda, newx=X[test,])
    val.MSE.en <- mean((pred.elastic - y[test])^2)
    
    # Re-calculate the model on all of the data.
    out <- glmnet(X,y,alpha=0, lambda=bestlambda)
    en.coef <- predict(out, type="coefficients")
    en.coef
    
    # go with elastic net - lasso can zero out variables and ridge seems more sensitive to early weeks
    weights <- en.coef
    
    # Apply weights to dataset
    current$power_rank_score_w <- (current$win_index * weights[2]) - ((current$luck_index) * weights[3]) + (current$consistency_index * weights[4]) + (current$score_index * weights[5]) + (current$season_index * weights[6])
    
    # Caolculate weighed power ranks
    current <- current %>% 
        group_by(week) %>% 
        mutate(power_rank_avg_w = mean(power_rank_score_w), power_score_w = round((power_rank_score_w / power_rank_avg_w) * 100),
               rank_w = (league_size + 1) - rank(power_rank_score_w)) %>% 
        arrange(team, week) %>% 
        group_by(team) %>% 
        mutate(rank_w_change = -(rank_w - lag(rank_w)), 
               score_w_change = power_score_w - lag(power_score_w))
    
    return(current)
}

get_pr_table <- function(current){
    # Create Summary Table
    table <- current[current$week == max(current$week),][c('team','rank_w','rank_w_change','power_score_w','score_w_change')]
    table <- arrange(table, -desc(rank_w))
    
    # set colors
    custom_blue <- "#7f9dff"
    custom_red <- '#ff7f7f'
    fixed_width = 50
    
    # format signs
    sign_format <- formatter('span', 
                             style = x ~ style(color = ifelse(x > 0, custom_blue, 
                                                              ifelse(x < 0, custom_red, 'blank'))),
                             x ~ icontext(ifelse(x > 0, 'arrow-up', 
                                                 ifelse(x < 0, 'arrow-down', x)), x))
    
    
    # Make the table
    final_table <- formattable(table, 
                               align=c('l', rep('c', 4)),
                               col.names = c("Team", "Rank", "1 Wk Change", "Score", "1 Wk Change"), 
                               list('rank_w_change' = sign_format,
                                    'score_w_change' = sign_format,
                                    style(width = paste(fixed_width), 'px', sep='')))
    return(final_table)
}

get_pr_rank <- function(current){
    # Plot weekly Power Ranking
    rank_chart <- current %>% 
        ggplot(aes(x=week, y=rank_w, col=team)) +
        geom_line(aes(linetype = team), size=1.05) +
        scale_y_continuous(breaks = seq(1,10,1), trans = "reverse") +
        scale_x_continuous(breaks = seq(1,max(current$week),1), limits = c(1,max(current$week)+0.1)) +
        theme_bw() +
        geom_text(aes(label=team, size=9), data=current[current$week == max(current$week),], hjust=-0.18) +
        xlab('Week') + ylab('Rank') +
        theme(legend.position = 'none',
              plot.title = element_text(size=18, face="bold",hjust = 0.5),
              panel.grid.minor = element_blank(),
              axis.text = element_text(size=11),
              axis.title = element_text(size=13),
              axis.ticks = element_blank()) +
        ggtitle(paste('Weekly Power Rankings'))
    
    return(rank_chart)
}

get_pr_score <- function(current){
    # Plot weekly Power Score
    score_chart <- current %>% 
        ggplot(aes(x=week, y=power_score_w, col=team)) +
        geom_line(aes(linetype = team), size=1.05) +
        geom_hline(yintercept = 100, linetype = 'dashed') +
        scale_x_continuous(breaks = seq(1,max(current$week),1), limits = c(1,max(current$week)+0.1)) +
        theme_bw() +
        geom_text_repel(aes(label=team, size=9), data=current[current$week == max(current$week),], nudge_x = 0.1) +
        xlab('Week') + ylab('Score') +
        theme(legend.position = 'none',
              plot.title = element_text(size=18, face="bold",hjust = 0.5),
              panel.grid.minor = element_blank(),
              axis.text = element_text(size=11),
              axis.title = element_text(size=13),
              axis.ticks = element_blank()) +
        ggtitle(paste('Weekly Power Score'))
    
    return(score_chart)
}

current <- get_weights()
get_pr_table(get_weights())
get_pr_rank(get_weights())
get_pr_score(get_weights())


# Simulations ----------------------------------------------------------------
# sim_season <- sim_season_table
# 
# if(py$matchup_week < py$regular_season_end){
#     results <- py$sim_season_results
#     results$avg_w <- round(results$avg_w, 1)
#     results$avg_pts <- round(results$avg_pts)
#     results$sd_w <- round(results$sd_w, 1)
#     results$sd_pts <- round(results$sd_pts)
#     results <- results[with(results, order(-avg_w, -avg_pts)),][c(1,3,2,4)]
# } else{
#     standings <- sim_season[,-c(2,3)]
#     standings$pfpg <- round(standings$score / (py$regular_season_end-1), 2)
#     standings$papg <- round(standings$pa / (py$regular_season_end-1), 2)
#     standings <- format(standings, big.mark = ',')
#     rownames(standings) <- standings$team
#     standings <- standings[,-1]
# }
# 
# sim_playoffs <- py$sim_playoffs
# sim_playoffs$prob_playoffs <- sim_playoffs$n_playoffs / py$n
# sim_playoffs$prob_finals <- sim_playoffs$n_finals / py$n
# sim_playoffs$prob_champ <- sim_playoffs$n_champ / py$n
# sim_playoffs$prob_second <- sim_playoffs$n_second / py$n
# sim_playoffs$prob_third <- sim_playoffs$n_third / py$n
# sim_playoffs$x.po <- (sim_playoffs$prob_champ * 350) + (sim_playoffs$prob_second * 100) + (sim_playoffs$prob_third * 50)
# sim_playoffs <- sim_playoffs[,c(6:11)]
# 
# # Format dataframe for table
# #sim_playoffs <- tibble::rownames_to_column(sim_playoffs, "team")
# sim_playoffs <- sim_playoffs[with(sim_playoffs, order(-x.po, -prob_champ, -prob_second, -prob_third, -prob_playoffs)), ]
# 
# accuracy <- .1
# sim_playoffs$prob_playoffs <- percent_format(accuracy = accuracy)(sim_playoffs$prob_playoffs)
# sim_playoffs$prob_finals <- percent_format(accuracy = accuracy)(sim_playoffs$prob_finals)
# sim_playoffs$prob_champ <- percent_format(accuracy = accuracy)(sim_playoffs$prob_champ)
# sim_playoffs$prob_second <- percent_format(accuracy = accuracy)(sim_playoffs$prob_second)
# sim_playoffs$prob_third <- percent_format(accuracy = accuracy)(sim_playoffs$prob_third)
# sim_playoffs$x.po <- dollar_format(accuracy = 1)(sim_playoffs$x.po)
# 
# # Make the tables
# if(py$matchup_week < py$regular_season_end){
#     season_table <- formattable(sim_season[,-ncol(sim_season)],
#                                 align = c(rep('c',ncol(sim_season)-1)),
#                                 col.names = c(seq(0,reg_season_end)))
# } else{
#     season_table <- formattable(standings,
#                                 col.names = c('Record', 'PF', 'PA', 'PF/G', 'PA/G'))
# }
# 
# if(py$matchup_week < py$regular_season_end){
#     results_table <- formattable(results,
#                                  align = c(rep('c', ncol(results))),
#                                  col.names = c('xWins', '+/-', 'xPF', '+/-'))
# }
# 
# playoff_table <- formattable(sim_playoffs,
#                              align = c(rep('c', ncol(sim_playoffs))),
#                              col.names = c('Playoffs', 'Finals', '1st', '2nd', '3rd', 'xPay Out'))



# Betting Table -------------------------------------------------------
# Calculate betting lines
# sim_week <- py$sim_week
# 
# # get avg scores
# sim_week_scores <- py$sim_week_scores
# avg_scores <- sim_week_scores %>%
#     group_by(team) %>% 
#     summarise(avg_score = round(mean(score), 1))
# 
# sim_week <- merge(sim_week, avg_scores, by='team')
# 
# # calculate game line
# sim_week$p_win <- sim_week$n_wins / py$n
# sim_week$betting_line <- ifelse(sim_week$p_win > 0.5,
#                                 (100*sim_week$p_win) / (1 - sim_week$p_win)*-1,
#                                 (100 / sim_week$p_win) - 100)
# sim_week$betting_line <- ifelse(is.infinite(sim_week$betting_line), 0, sim_week$betting_line)
# sim_week$betting_line <- plyr::round_any(sim_week$betting_line, 5)
# 
# # calculate high score line
# sim_week$p_high <- round(sim_week$n_highest / py$n,2)
# sim_week$high_line <- ifelse(sim_week$p_high > 0.5,
#                              (100*sim_week$p_high) / (1 - sim_week$p_high)*-1,
#                              (100 / sim_week$p_high) - 100)
# sim_week$high_line <- ifelse(is.infinite(sim_week$high_line), 0, sim_week$high_line)
# sim_week$high_line <- plyr::round_any(sim_week$high_line, 5)
# 
# # calculate low score line
# sim_week$p_low <- round(sim_week$n_lowest / py$n,2)
# sim_week$low_line <- ifelse(sim_week$p_low > 0.5,
#                             (100*sim_week$p_low) / (1 - sim_week$p_low)*-1,
#                             (100 / sim_week$p_low) - 100)
# sim_week$low_line <- ifelse(is.infinite(sim_week$low_line), 0, sim_week$low_line)
# sim_week$low_line <- plyr::round_any(sim_week$low_line, 5)
# 
# 
# sim_week <- sim_week[with(sim_week, order(game_id, p_win)), ]
# 
# accuracy <- .1
# sim_week$p_win <- percent_format(accuracy = accuracy)(sim_week$p_win)
# sim_week$betting_line <- ifelse(sim_week$betting_line == 0, '-', sprintf("%+3d", sim_week$betting_line))
# sim_week$high_line <- ifelse(sim_week$high_line == 0, '-', sprintf("%+3d", sim_week$high_line))
# sim_week$low_line <- ifelse(sim_week$low_line == 0, '-', sprintf("%+3d", sim_week$low_line))
# 
# rownames(sim_week) <- sim_week[,1]
# sim_week[,1] <- NULL
# 
# betting_table <- formattable(sim_week[,c('game_id', 'p_win', 'betting_line', 'avg_score', 'high_line', 'low_line')],
#                              align = c('c', 'c', 'c', 'c', 'c', 'c'),
#                              col.names = c('Matchup', 'Win %', 'Game Line', 'Proj. Score', 'High Scorer', 'Low Scorer'))



# Scenarios ---------------------------------------------------------------------
d <- load_data(league_id, season, swid="", espn="")
regular_season_end <- get_params(d)[[3]]
league_size <- get_params(d)[[1]]

scen <- scenarios(d, get_weights())

get_league_records <- function(scen){
    
    wins_vs_league <- scen[[1]]
    
    tot_wins <- wins_vs_league %>% 
        group_by(team) %>% 
        summarise(wins = sum(teams_beat),
                  win_perc = round(wins / (py$regular_season_end * (length(unique(wins_vs_league$team))-1)), 3)) %>% 
        arrange(desc(win_perc))
    tot_wins$matchups <- (league_size-1) * week
    tot_wins$record <- paste0(tot_wins$wins, "-", (tot_wins$matchups-tot_wins$wins))
    
    # how many wins would the team on the left have if they played the team on the top every week?
    wins_vs_opp <- scen[[3]]
    wins_vs_opp$team <- rownames(wins_vs_opp)
    
    # convert wide to long to calculate record
    df <- reshape(wins_vs_opp,
                  direction = "long",
                  varying = list(names(wins_vs_opp)[1:league_size]),
                  v.names = "wins",
                  idvar = "team")
    df$losses <- week - df$wins
    df$record <- paste0(df$wins, "-", df$losses)
    
    # convert back to wide to get final table
    record_table <- df %>% select(team, time, record)
    record_table <- reshape(record_table, 
                            idvar = "team",
                            timevar = "time",
                            direction = "wide")
    colnames(record_table)[2:length(colnames(record_table))] <- record_table$team
    record_table$id <- 1:nrow(record_table)
    rownames(record_table) <- record_table$team
    
    record_table <- merge(record_table, tot_wins[,c(1,5,3)], by = "team") %>% 
        arrange(desc(win_perc))
    record_table <- record_table[,c(c(record_table$id)+1, ncol(record_table)-1, ncol(record_table), 1)]
    record_table <- record_table %>%  
        rename(Record = record,
               'Win%' = win_perc)
    
    # replace diagonals with blanks
    mat <- as.matrix(record_table)
    diag(mat) <- ""
    record_table_new <- data.frame(mat)
    colnames(record_table_new) <- colnames(record_table)
    #record_table_new$team <- rownames(record_table_new)
    rownames(record_table_new) <- NULL
    colnames(record_table_new)[ncol(record_table_new)] <- " "
    
    record_vs_opp_table <- formattable(record_table_new[,c(ncol(record_table_new),seq(1,ncol(record_table_new)-1))],
                                       align = c("l", rep("c", ncol(record_table_new)-1)),
                                       list(~ formatter("span",
                                                        style = x ~ style("width" = "20px")),
                                            " " = formatter("span",style = ~ style("font.weight" = "bold",  
                                                                                   "text-align" = "left"))))
    
    return(record_vs_opp_table)
}

get_schedule_swich <- function(scen){
    switched_sched <- scen[[2]]

    # Scenario 2: Season record given another team's schedule
    switched_sched$team.sched <- rownames(switched_sched)
    act_wins <- as.numeric(diag(as.matrix(switched_sched)))
    
    # convert df to long, get record, and find win differentials
    df <- reshape(switched_sched,
                  direction = "long",
                  varying = list(names(switched_sched)[1:league_size]),
                  v.names = "wins",
                  idvar = "team.sched")
    
    # return record
    df$losses <- py$regular_season_end - df$wins
    df$record <- paste0(df$wins, "-", df$losses)
    
    # difference from actual wins
    df$team.new.sched <- rep(switched_sched$team.sched, each=length(league_size))
    df$act_wins <- rep(act_wins, each=league_size)
    df$win_diff <- df$wins - df$act_wins
    df$record_win_diff <- paste0(df$record, 
                                 " (", 
                                 ifelse(df$win_diff >= 0, paste0("+", df$win_diff), df$win_diff), 
                                 ")")
    
    record_table <- df %>% select(team.sched, time, record_win_diff)
    record_table <- reshape(record_table, 
                            idvar = "team.sched",
                            timevar = "time",
                            direction = "wide")
    rownames(record_table) <- record_table$team.sched
    colnames(record_table)[2:length(colnames(record_table))] <- record_table$team.sched
    
    # replace diagonals with blanks
    mat <- as.matrix(record_table)
    diag(mat[,2:c(ncol(mat))]) <- ""
    record_table <- data.frame(mat)
    colnames(record_table)[2:ncol(record_table)] <- rownames(record_table)
    colnames(record_table)[1] <- " "
    rownames(record_table) <- 1:nrow(record_table)
    
    # how many wins would the team on the top have if they had the team's schedule on the left?
    record_table_final <- formattable(record_table, 
                                      align = c("l", rep("c", ncol(record_table)-1)),
                                      list(~ formatter("span",
                                                       style = x ~ style("width" = "20px")),
                                           " " = formatter("span",style = ~ style("font.weight" = "bold"))))
    
    return(record_table_final)
}


# Efficiencies ---------------------------------------------------------------------
get_eff <- function(){
    eff <- df_scores
    colnames(eff) <- gsub(x = names(eff), pattern = "/", replacement = ".") 
    
    # add team and positional efficiency scores
    eff$team.eff <- 1 - ((eff$opts - eff$apts) / eff$opts)
    eff$qb.eff <- 1 - ((eff$QB.opts - eff$QB.apts) / eff$QB.opts)
    eff$rb.eff <- 1 - ((eff$RB.opts - eff$RB.apts) / eff$RB.opts)
    eff$wr.eff <- 1 - ((eff$WR.opts - eff$WR.apts) / eff$WR.opts)
    eff$te.eff <- 1 - ((eff$TE.opts - eff$TE.apts) / eff$TE.opts)
    eff$dst.eff <- 1 - ((eff$D.ST.opts - eff$D.ST.apts) / eff$D.ST.opts)
    eff$k.eff <- 1 - ((eff$K.opts - eff$K.apts) / eff$K.opts)
    
    # combine RB/WR/TE into overall flex
    # need to use actual flex positions
    eff$Flex.opts <- eff$RB.opts + eff$WR.opts + eff$TE.opts + eff$Flex.opts
    eff$Flex.apts <- eff$RB.apts + eff$WR.apts + eff$TE.apts + eff$Flex.apts
    eff$Flex.eff <- 1 - ((eff$Flex.opts - eff$Flex.apts) / eff$Flex.opts)
    
    # replace nan's with 1 because that is still the most efficient outcome
    eff <- eff %>% replace(is.na(.), 1)
    
    # replace inf with 0 - best player scored 0, but actual starter scored less
    eff[sapply(eff, is.infinite)] <- 0
    
    # if efficiency is >1, take the inverse
    eff[,19:ncol(eff)][eff[,19:ncol(eff)] > 1] <- 1 / eff[,19:ncol(eff)][eff[,19:ncol(eff)] > 1]
    
    colnames(eff)[c(1,2)] <- c("Team.opts", "Team.apts")
    
    return(eff)
}

get_optimal <- function(eff, pos = c("Team", "QB", "RB", "WR", "TE", "Flex", "D.ST", "K")){
    ### Optimal vs Actual
    cols <- eff %>% dplyr::select(team, week, starts_with(pos)) %>% colnames()
    
    eff2 <- eff %>% 
        select(cols) %>% 
        rename_all(~str_replace_all(., "^.*\\.", "")) %>% 
        group_by(team) %>% 
        summarise(act = sum(apts),
                  opt = sum(opts),
                  opt.per.week = opt / max(week),
                  act.per.week = act / max(week),
                  diff.per.week = act.per.week - opt.per.week,
                  effic = act / opt)
    
    eff_plot <- eff2 %>% 
        ggplot(aes(x = diff.per.week, y = opt.per.week, col = team)) +
        geom_point() +
        geom_vline(xintercept = mean(eff2$diff.per.week), col = "#999999") +
        geom_hline(yintercept = mean(eff2$opt.per.week), col = "#999999") +
        geom_text_repel(aes(label = paste0(team, " ", round(effic,2)*100, "%")), size = 3, vjust = 1.5) +
        xlim(min(eff2$diff.per.week)-2, max(eff2$diff.per.week)+2) +
        ylim(min(eff2$opt.per.week)-5, max(eff2$opt.per.week)+5) +
        annotate("text", 
                 x = min(eff2$diff.per.week)-0.5,
                 y = min(eff2$opt.per.week)-5,
                 label = "Bad roster, bad starts",
                 size = 3) +
        annotate("text",
                 x = min(eff2$diff.per.week)-0.5,
                 y = max(eff2$opt.per.week)+5,
                 label = "Good roster, bad starts",
                 size = 3) +
        annotate("text",
                 x = max(eff2$diff.per.week)+0.5,
                 y = max(eff2$opt.per.week)+5,
                 label = "Good roster, good starts",
                 size = 3) +
        annotate("text",
                 x = max(eff2$diff.per.week)+0.5,
                 y = min(eff2$opt.per.week)-5,
                 label = "Bad roster, good starts",
                 size = 3) +
        xlab("Difference from Optimal per Week") + ylab("Optimal Points per Week") + ggtitle(pos) +
        theme_bw() +
        theme(legend.position = "none",
              plot.title = element_text(size=14, face="bold",hjust = 0.5))
    
    return(eff_plot)
}

### Team efficiency by position
get_pos_eff <- function(eff){
    eff3 <- eff %>% 
        group_by(tm = team) %>% 
        summarise(team = mean(team.eff),
                  qb = mean(qb.eff),
                  rb = mean(rb.eff),
                  wr = mean(wr.eff),
                  te = mean(te.eff),
                  flex = mean(Flex.eff),
                  dst = mean(dst.eff),
                  k = mean(k.eff))
    
    eff3.long <- melt(as.data.table(eff3),
                      id.vars = 1,
                      measure.vars = 2:ncol(eff3),
                      variable.name = "position",
                      value.name = "efficiency")
    
    eff_by_pos <- eff3.long[eff3.long$position != "team"] %>% 
        ggplot(aes(x = efficiency, y = fct_rev(as_factor(tm)), col = position)) +
        geom_point() +
        geom_vline(xintercept = mean(eff3$team), col = "#999999") +
        xlab("Efficiency") + ylab("Team") +
        theme_bw() +
        scale_fill_brewer(palette = "Pastel2")
    
    return(eff_by_pos)
}


# Shiny App ---------------------------------------------------------------
library(shiny)
library(shinythemes)
ui <- fluidPage(theme = shinytheme("flatly"),
                titlePanel("Analyze Your Fantasy Football League!"),
                navbarPage("",
                           tabPanel("Setup",
                                    textInput(inputId = "league_id", label = "Enter League ID"),
                                    selectInput(inputId = "season", label = "Choose a Season", choices = 2018:2020),
                                    p("Add next two sections ONLY if your league is private:"),
                                    textInput(inputId = "swid", label = "SWID"),
                                    textInput(inputId = "espn", label = "ESPN_S2")),
                           tabPanel("Power Ranking",
                                    tags$div("Power rankings take into account 5 factors:", tags$br(),
                                      "1. Winning percentage", tags$br(),
                                      "2. Average score over the past 3 weeks compared to your league", tags$br(),
                                      "3. Total points for compared to your league", tags$br(),
                                      "4. Scoring consistency, with higher scores weighed more", tags$br(),
                                      "5. Actual matchup result compared to how many teams you 'would have' beaten")),
                           tabPanel("Simulation",
                                    p("Simulations are ran 1000 times")),
                           tabPanel("Betting Table",
                                    p("Odds are calculated based on current lineups")),
                           tabPanel("Scenarios"),
                           tabPanel("Team Efficiency")
                           )
                )

server <- function(input, output){
    
}

shinyApp(ui, server)
