//! Format-string interpolation for output file naming.
//!
//! Tokens are delimited by `[...]`. Unknown tokens are left verbatim.

/// Which output file type is being named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFileType {
    Color,
    Motion,
    Meta,
}

/// Resolved values for every token in the format string.
pub struct NameTokens<'a> {
    pub basename: &'a str,
    pub cols: u32,
    pub rows: u32,
    pub type_label: &'a str,
    pub ext: &'a str,
}

static DEFAULT_FORMAT: &str = "[basename]_[cols]x[rows][type].[ext]";

/// Interpolate a format string with the given token values.
///
/// If `format` is empty, [`DEFAULT_FORMAT`] is used.
/// Unknown tokens (e.g. `[foo]`) are left verbatim.
pub fn interpolate_name_format(format: &str, tokens: &NameTokens<'_>) -> String {
    let format = if format.is_empty() { DEFAULT_FORMAT } else { format };
    let mut result = String::with_capacity(format.len());
    let mut rest = format;
    while let Some(open) = rest.find('[') {
        result.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        if let Some(close) = rest.find(']') {
            let token = &rest[..close];
            match token {
                "basename" => result.push_str(tokens.basename),
                "cols" => result.push_str(&tokens.cols.to_string()),
                "rows" => result.push_str(&tokens.rows.to_string()),
                "type" => result.push_str(tokens.type_label),
                "ext" => result.push_str(tokens.ext),
                _ => {
                    result.push('[');
                    result.push_str(token);
                    result.push(']');
                }
            }
            rest = &rest[close + 1..];
        } else {
            result.push('[');
            result.push_str(rest);
            rest = "";
        }
    }
    result.push_str(rest);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_all_tokens() {
        let tokens = NameTokens {
            basename: "explosion".into(),
            rows: 4,
            cols: 8,
            type_label: "_MV",
            ext: "tga",
        };
        let result = interpolate_name_format("[basename]_[cols]x[rows][type].[ext]", &tokens);
        assert_eq!(result, "explosion_8x4_MV.tga");
    }

    #[test]
    fn unknown_tokens_left_verbatim() {
        let tokens = NameTokens {
            basename: "x".into(), rows: 1, cols: 1, type_label: "", ext: "tga",
        };
        let result = interpolate_name_format("[basename]_[foo].[ext]", &tokens);
        assert_eq!(result, "x_[foo].tga");
    }

    #[test]
    fn empty_format_falls_back_to_default() {
        let tokens = NameTokens {
            basename: "x".into(), rows: 3, cols: 4, type_label: "_meta", ext: "json",
        };
        let result = interpolate_name_format("", &tokens);
        assert_eq!(result, "x_4x3_meta.json");
    }

    #[test]
    fn empty_type_label_no_extra_separator() {
        let tokens = NameTokens {
            basename: "x".into(), rows: 2, cols: 2, type_label: "", ext: "tga",
        };
        let result = interpolate_name_format("[basename]_[cols]x[rows][type].[ext]", &tokens);
        assert_eq!(result, "x_2x2.tga");
    }

    #[test]
    fn custom_basename_overrides() {
        let tokens = NameTokens {
            basename: "my_custom_name".into(), rows: 1, cols: 1, type_label: "", ext: "tga",
        };
        let result = interpolate_name_format("[basename].[ext]", &tokens);
        assert_eq!(result, "my_custom_name.tga");
    }
}
