# Aspect-Based Sentiment Analysis of online restaurant reviews with VADER
# By: Joe Domaleski 
# More information about this R script on https://blog.marketingdatascience.ai

# Step 0 - Setup the environment 
# Load essential libraries
library(stopwords)   # Provides common stop words for text filtering
library(tidyverse)   # Data manipulation and visualization
library(vader)       # VADER sentiment analysis for text
library(wordcloud)   # Creates word clouds for frequency visualization
library(scales)      # Adjusts scales and labels in plots
library(tidytext)    # Tidy text processing for tokenization and word analysis

# Get ready for processing
rm(list = ls()) # Clear environment
set.seed(77) # Set seed for reproducibility

# Define specific stop words, removing "s" from the list and handling it separately
custom_stopwords <- c("a", "an", "the", "and", "or", "in", "on", "at", 
                      "of", "to", "from", "by", "with", "i", "you", 
                      "he", "she", "we", "they", "it", "my", "your", 
                      "his", "her", "our", "their", "its", "then", 
                      "when", "where", "there")
setwd("posts/asba-llm/")
# Step 1 - Load data and review it
reviews <- read_csv("yelp_reviews_bouchon_bakery_las_vegas.csv", show_col_types = FALSE) %>%
  sample_n(1000) %>%
  mutate(word_count = str_count(review, "\\w+")) %>%  # Calculate word count
  mutate(review = str_replace_all(review, paste(custom_stopwords, collapse = "\\b|\\b"), "")) %>%
  mutate(review = str_replace_all(review, "\\bs\\b", "")) %>%  # Remove standalone "s"
  mutate(review = str_replace_all(review, "\\bice cream\\b", "icecream"))  # Treat "ice cream" as a single word
# Word Frequency Distribution for Initial Exploration
initial_word_frequencies <- reviews %>%
  unnest_tokens(word, review) %>%
  filter(!word %in% custom_stopwords) %>%
  count(word, sort = TRUE) %>%
  filter(n > 1)  # Exclude single occurrences for clarity

# Filter to keep only the top 30 most frequent words
top_words <- initial_word_frequencies %>%
  slice_max(n, n = 30)

# Plot the Top Word Frequency Distribution
plot_top_word_freq <- ggplot(top_words, aes(x = reorder(word, n), y = n)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Top 30 Word Frequency Distribution (Stop Words Removed)", x = "Words", y = "Frequency") +
  theme_minimal()
print(plot_top_word_freq)

# Rating Distribution Plot
plot_rating_distribution <- ggplot(reviews, aes(x = as.factor(rating))) +
  geom_bar(fill = "skyblue") +
  labs(title = "Distribution of Ratings", x = "Rating", y = "Count") +
  theme_minimal()
print(plot_rating_distribution)

# Step 2: Define the aspects and keywords for each aspect
aspects <- list(
  food = c("food", "taste", "icecream", "flavors", "chocolate", "try", "pastry", "bread", "dessert"),
  service = c("service", "staff", "waiter", "waitress", "get", "back", "host", "manager"),
  ambience = c("ambience", "atmosphere", "decor", "music", "environment", "place", "setting"),
  price = c("price", "cost", "expensive", "cheap", "value", "worth")
)

# Step 3: Define a function to detect aspect-related sentences and apply VADER sentiment analysis
get_aspect_sentiment <- function(review_text) {
  aspect_sentiments <- tibble(
    aspect = character(),
    compound = double(),
    pos = double(),
    neu = double(),
    neg = double(),
    words = list()
  )
  
  for (aspect in names(aspects)) {
    keywords <- aspects[[aspect]]
    aspect_sentences <- review_text[grepl(paste(keywords, collapse = "|"), review_text, ignore.case = TRUE)]
    
    if (length(aspect_sentences) > 0) {
      sentiment <- as.list(get_vader(paste(aspect_sentences, collapse = ". ")))
      words_list <- list(strsplit(paste(aspect_sentences, collapse = " "), " ")[[1]])
      
      if (all(c("compound", "pos", "neu", "neg") %in% names(sentiment))) {
        aspect_sentiments <- aspect_sentiments %>%
          add_row(
            aspect = aspect,
            compound = as.numeric(sentiment$compound),
            pos = as.numeric(sentiment$pos),
            neu = as.numeric(sentiment$neu),
            neg = as.numeric(sentiment$neg),
            words = words_list
          )
      }
    }
  }
  
  return(aspect_sentiments)
}

## TODO: do this on full data
# for now restrict to first 100
reviews <- reviews[1:100,]

# Step 4: Apply the function to each review and create a results table
aspect_sentiments <- reviews %>%
  rowwise() %>%
  mutate(aspect_sentiment = list({
    cat("Processing row:", cur_group_id(), "\n")
    get_aspect_sentiment(review)
  })) %>%
  unnest(aspect_sentiment) %>%
  mutate(original_review = review, rating = rating, word_count = word_count)  # Add rating and word count

# Display results table
aspect_summary <- aspect_sentiments %>%
  group_by(aspect) %>%
  summarise(
    avg_compound = mean(compound, na.rm = TRUE),
    avg_pos = mean(pos, na.rm = TRUE),
    avg_neu = mean(neu, na.rm = TRUE),
    avg_neg = mean(neg, na.rm = TRUE),
    count = n()
  )
print(aspect_summary)

# Correlation analysis between rating and compound sentiment
correlation <- cor(as.numeric(aspect_sentiments$rating), aspect_sentiments$compound, use = "complete.obs")
cat("Correlation between Rating and Compound Sentiment:", round(correlation, 3), "\n")

# Group by original review and concatenate aspects and sentiment scores in a single row
grouped_aspect_scores <- aspect_sentiments %>%
  group_by(original_review) %>%
  summarise(
    aspects = paste(aspect, collapse = ", "),
    compound_scores = paste(round(compound, 3), collapse = "; "),
    positive_scores = paste(round(pos, 3), collapse = "; "),
    neutral_scores = paste(round(neu, 3), collapse = "; "),
    negative_scores = paste(round(neg, 3), collapse = "; ")
  ) %>%
  slice_head(n = 10)  # Display the first 10 rows for brevity

# Print the grouped output
print(grouped_aspect_scores)

# Step 5: Generate Plots

# Plot 1: Density of compound sentiment scores by aspect
# Adjusted Density Plot for Compound Sentiment Scores by Aspect
ggplot(aspect_sentiments %>% filter(is.finite(compound)), aes(x = compound, fill = aspect)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~aspect, scales = "free_y") +
  labs(
    title = "Density of Compound Sentiment Scores by Aspect",
    x = "Compound Sentiment Score (VADER)",
    y = "Density (Frequency of Sentiment)"
  ) +
  scale_x_continuous(breaks = seq(-1, 1, by = 0.5), limits = c(-1, 1)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.6) +  # Neutral reference line
  scale_fill_manual(values = c("ambience" = "pink", "food" = "lightgreen", "price" = "skyblue", "service" = "mediumpurple")) +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Plot 2: Compound Sentiment by Rating (Sanity Check)

# Filter out non-finite values and add descriptive labels for ratings
aspect_sentiments_filtered <- aspect_sentiments %>%
  filter(is.finite(compound)) %>%
  mutate(rating_label = factor(rating, 
                               levels = 1:5, 
                               labels = c("Very Negative (1)", "Negative (2)", "Neutral (3)", "Positive (4)", "Very Positive (5)")))

plot_rating_vs_sentiment <- ggplot(aspect_sentiments_filtered, aes(x = rating_label, y = compound)) +
  geom_boxplot(aes(fill = rating_label)) +
  labs(title = "Compound Sentiment Score by Rating", x = "Rating", y = "Compound Sentiment Score") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability
print(plot_rating_vs_sentiment)

# Plot 3: Compound Sentiment by Aspect
plot_compound_sentiment <- ggplot(aspect_sentiments %>% filter(!is.na(compound)), aes(x = aspect, y = compound, fill = compound > 0)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "red"), labels = c("Positive", "Negative")) +
  labs(title = "Aspect-Based Sentiment Scores", x = "Aspect", y = "Compound Sentiment", fill = "Sentiment") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(plot_compound_sentiment)

# Plot 4: Proportions of Positive, Neutral, and Negative Sentiment by Aspect
plot_aspect_sentiments <- aspect_sentiments %>%
  group_by(aspect) %>%
  summarise(
    pos_pct = mean(pos, na.rm = TRUE) * 100,
    neu_pct = mean(neu, na.rm = TRUE) * 100,
    neg_pct = mean(neg, na.rm = TRUE) * 100
  ) %>%
  pivot_longer(cols = c(pos_pct, neu_pct, neg_pct), names_to = "sentiment", values_to = "percentage") %>%
  mutate(sentiment = factor(sentiment, levels = c("pos_pct", "neu_pct", "neg_pct"), labels = c("Positive", "Neutral", "Negative"))) %>%
  ggplot(aes(x = aspect, y = percentage, fill = sentiment)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Positive" = "blue", "Neutral" = "gray", "Negative" = "red")) +
  labs(title = "Sentiment Composition by Aspect", x = "Aspect", y = "Percentage", fill = "Sentiment Type") +
  theme_minimal()
print(plot_aspect_sentiments)

# Plot 5: Aspect Frequency Distribution
plot_aspect_freq <- ggplot(aspect_summary, aes(x = aspect, y = count, fill = aspect)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Aspect Frequency Distribution", x = "Aspect", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")
print(plot_aspect_freq)

# Plot 6: Violin Plot: Compound Score Distribution by Aspect
plot_violin <- ggplot(aspect_sentiments %>% filter(is.finite(compound)), aes(x = aspect, y = compound, fill = aspect)) +
  geom_violin(trim = FALSE, color = "black") +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.5) +  # Add a boxplot inside the violin for additional detail
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Compound Score Distribution by Aspect (Violin Plot)", x = "Aspect", y = "Compound Sentiment") +
  theme_minimal() +
  theme(legend.position = "none")
print(plot_violin)

# Step 6: Create combined word clouds for positive and negative words by aspect
par(mfrow = c(2, 2), mar = c(1, 1, 1, 1))  # Set up a 2x2 plotting grid, adjust margins

for (aspect_name in unique(aspect_sentiments$aspect)) {
  aspect_words <- aspect_sentiments %>% filter(aspect == aspect_name)
  
  # Process words for positive and negative sentiment
  positive_words <- suppressWarnings(unlist(aspect_words %>% filter(compound > 0) %>% pull(words)) %>%
                                       str_replace_all("[[:punct:]]", "") %>%
                                       tolower() %>%
                                       .[!. %in% stopwords::stopwords("en")])
  
  negative_words <- suppressWarnings(unlist(aspect_words %>% filter(compound < 0) %>% pull(words)) %>%
                                       str_replace_all("[[:punct:]]", "") %>%
                                       tolower() %>%
                                       .[!. %in% stopwords::stopwords("en")])
  
  # Positive word cloud
  if (length(positive_words) > 0) {
    suppressWarnings(
      wordcloud(words = positive_words, scale = c(1.8, 0.4), max.words = 100, colors = "blue")
    )
    title(main = paste("Positive:", aspect_name), cex.main = 0.7)
  } else {
    message(paste("No positive words found for aspect:", aspect_name))
  }
  
  # Negative word cloud
  if (length(negative_words) > 0) {
    suppressWarnings(
      wordcloud(words = negative_words, scale = c(1.8, 0.4), max.words = 100, colors = "red")
    )
    title(main = paste("Negative:", aspect_name), cex.main = 0.7)
  } else {
    message(paste("No negative words found for aspect:", aspect_name))
  }
}

# Reset par to default for future plots
par(mfrow = c(1, 1))

# Step 7: Generate Summary Insights for Each Aspect
summary_insights <- function(aspect_data) {
  aspect <- unique(aspect_data$aspect)
  
  avg_compound <- mean(aspect_data$compound, na.rm = TRUE)
  most_positive <- aspect_data %>%
    filter(compound == max(compound, na.rm = TRUE)) %>%
    pull(original_review) %>%
    head(1)
  
  most_negative <- aspect_data %>%
    filter(compound == min(compound, na.rm = TRUE)) %>%
    pull(original_review) %>%
    head(1)
  
  common_positive_words <- aspect_data %>%
    filter(compound > 0) %>%
    pull(words) %>%
    unlist() %>%
    str_replace_all("[[:punct:]]", "") %>%
    tolower() %>%
    .[. != "" & !. %in% stopwords::stopwords("en")] %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    head(5) %>%
    names()
  
  common_negative_words <- aspect_data %>%
    filter(compound < 0) %>%
    pull(words) %>%
    unlist() %>%
    str_replace_all("[[:punct:]]", "") %>%
    tolower() %>%
    .[. != "" & !. %in% stopwords::stopwords("en")] %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    head(5) %>%
    names()
  
  cat("Summary for Aspect:", aspect, "\n")
  cat("Average Compound Score:", round(avg_compound, 3), "\n")
  cat("Most Positive Review:", most_positive, "\n")
  cat("Most Negative Review:", most_negative, "\n")
  cat("Common Positive Words:", paste(common_positive_words, collapse = ", "), "\n")
  cat("Common Negative Words:", paste(common_negative_words, collapse = ", "), "\n\n")
}

# Apply the summary function to each aspect
cat("Key Takeaways from Aspect-Based Sentiment Analysis:\n\n")
for (aspect_name in unique(aspect_sentiments$aspect)) {
  aspect_data <- aspect_sentiments %>% filter(aspect == aspect_name)
  summary_insights(aspect_data)
}

